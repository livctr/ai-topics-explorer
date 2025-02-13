from db_utils import return_conn

import os
import json
import tempfile
import psycopg2
import openai
import asyncio
import json
import glob
import json


from openai import OpenAI


def requestify_keyword_extraction(arxiv_id: str,
                                  abstract: str,
                                  model: str = "gpt-4o-mini",
                                  temperature: float = 0.0,
                                  max_tokens: int = 100,
                                  max_keywords: int = 7,
):
    """Prepares abstracts for batch keyword extraction (cheaper).
    
    Nothing special. See https://cookbook.openai.com/examples/batch_processing
    """
    prompt = (
        f"Extract at most {str(max_keywords)} keywords from the arXiv abstract below, comma-separated. "
        "These keywords should indicate what the paper is about (e.g., topic, problem, method, solution).\n\n"
    )

    task = {
        "custom_id": arxiv_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "developer",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": abstract
                }

            ],
        }
    }
    return task


def extract_keyword(arxiv_id: str, abstract: str):
    """Tests `requestify_keyword_extraction."""
    req = requestify_keyword_extraction(arxiv_id, abstract)
    print(req)
    client = OpenAI()
    completion = client.chat.completions.create(**req['body'])
    return completion.choices[0].message.content


# Helper function to poll for job completion.
async def poll_job(client, batch_idx, batch_id):
    await asyncio.sleep(30)
    cntr = 0
    while True:
        batch_job = client.batches.retrieve(batch_id)

        if batch_job.status == "completed" and batch_job.output_file_id is not None:
            result_file_id = batch_job.output_file_id
            result = client.files.content(result_file_id).content
            result_file_name = f"data/tmp/keyword-output-{str(batch_idx).zfill(5)}.jsonl"
            with open(result_file_name, "wb") as f:
                f.write(result)
        else:
            cntr += 1
            if cntr % 5 == 0:
                print(f"Waiting on batch {str(batch_idx)} (~{cntr} mins): {batch_job}")
            await asyncio.sleep(60)



async def extract_keywords_batch(
        max_requests: int = 50,
        requests_batch_size: int = 2,
        max_keywords: int = 7
):
    # Get abstracts from the DB.
    conn = return_conn()
    cur = conn.cursor()
    if max_requests is not None:
        cur.execute("SELECT arxiv_id, abstract FROM paper WHERE keywords IS NULL LIMIT %s", (max_requests,))
    else:
        cur.execute("SELECT arxiv_id, abstract FROM paper WHERE keywords IS NULL")
    papers = cur.fetchall()
    cur.close()
    conn.close()

    # Build a list of prompt requests.
    num_chunks = len(papers) // requests_batch_size + (1 if len(papers) % requests_batch_size != 0 else 0)
    batch_jobs = []

    # init client
    client = OpenAI()

    for i in range(num_chunks):
        chunk = papers[i * requests_batch_size:(i + 1) * requests_batch_size]

        # Write the chunk to a temporary file.
        input_filename = f"data/tmp/keyword-input-{str(i).zfill(5)}.jsonl"
        with open(input_filename, "w") as f:
            for arxiv_id, abstract in chunk:
                req = requestify_keyword_extraction(arxiv_id, abstract, max_keywords=max_keywords)
                f.write(json.dumps(req) + "\n")
        
        # Upload batch file
        batch_file = client.files.create(
            file=open(input_filename, "rb"),
            purpose="batch"
        )

        # Create batch job
        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"Submitting batch {i + 1}/{num_chunks}.")
        print(f" - {batch_file}")
        print(f" - {batch_job}")

        # Poll for job completion
        batch_jobs.append(
            asyncio.create_task(poll_job(client, i, batch_job.id))
        )

    await asyncio.gather(*batch_jobs)



def send_keywords_to_db(keywords_list):
    """
    Given a list of tuples (arxiv_id, keywords), update the 'paper' table so that
    each record's keywords attribute is set accordingly.
    
    Parameters:
        keywords_list (list of tuple): Each tuple is (arxiv_id, keywords)
    """
    conn = return_conn()
    try:
        with conn:
            cur = conn.cursor()
            # Note: the input tuple is (arxiv_id, keywords), so we need to reverse the order.
            query = "UPDATE paper SET keywords = ? WHERE arxiv_id = ?"
            params = [(keywords, arxiv_id) for arxiv_id, keywords in keywords_list]
            cur.executemany(query, params)
    except Exception as e:
        print("Error updating keywords in the database:", e)
    finally:
        conn.close()


def send_keywords_to_db():
    """
    Reads all files matching the pattern:
        data/tmp/keyword-output-<5-digit-number>.jsonl

    The function verifies that each entry is successful (status code 200, error is null, and keywords exists).
    It then updates the `paper` table so that for each paper with matching arxiv_id, the `keywords` column is set.
    Returns True if every line was successful and the update completes, otherwise returns False.
    """
    # Collect tuples of (keywords, arxiv_id) for the database update.
    updates = []
    # Use glob to find all matching files.
    file_pattern = "data/tmp/keyword-output-*.jsonl"
    files = glob.glob(file_pattern)
    
    # Process each file.
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue  # skip empty lines
                    try:
                        output_entry = json.loads(line)
                    except Exception as e:
                        print(f"Error parsing JSON in file {file_path} at line {line_number}: {e}")
                        return False
                    
                    # Check if there's any error.
                    if output_entry.get("error") is not None:
                        print(f"Error found in entry in file {file_path} at line {line_number}: {output_entry.get('error')}")
                        return False
                    
                    # Check that the response has a 200 status.
                    response = output_entry.get("response")
                    if response is None or response.get("status_code") != 200:
                        print(f"Non-success status code in file {file_path} at line {line_number}")
                        return False
                    
                    # Extract the keywords from the response.
                    try:
                        keywords = response["body"]["choices"][0]["message"]["content"]
                    except (KeyError, IndexError) as e:
                        print(f"Error extracting keywords in file {file_path} at line {line_number}: {e}")
                        return False
                    if not keywords:
                        print(f"Empty keywords in file {file_path} at line {line_number}")
                        return False
                    
                    # Extract the arxiv_id from "custom_id"
                    arxiv_id = output_entry.get("custom_id")
                    if not arxiv_id:
                        print(f"Missing custom_id in file {file_path} at line {line_number}")
                        return False
                    
                    # Append tuple in order for the parameterized query: (keywords, arxiv_id)
                    updates.append((keywords, arxiv_id))
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return False

    # If no updates were collected, decide if that is success or not.
    # Here, we assume that having no files or no entries is a failure.
    if not updates:
        print("No valid keyword entries were found.")
        return False

    # Update the database.
    conn = return_conn()
    try:
        with conn:
            cur = conn.cursor()
            query = "UPDATE paper SET keywords = ? WHERE arxiv_id = ?"
            cur.executemany(query, updates)
        return True
    except Exception as e:
        print("Error updating keywords in the database:", e)
        return False
    finally:
        conn.close()


# To run the async function:
if __name__ == "__main__":
    # asyncio.run(extract_keywords_batch(max_requests=5, requests_batch_size=5, max_keywords=7))

    # send_keywords_to_db()

    pass
