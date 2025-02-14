import glob
import os
import json
import warnings

import asyncio
from openai import OpenAI

from db_utils import return_conn
from core.classify import requestify_keyword_extraction


def requestify_keyword_extraction_batch(*, arxiv_id: str, **kwargs):
    """Prepares abstracts for batch keyword extraction (cheaper).
    
    Nothing special. See https://cookbook.openai.com/examples/batch_processing
    """

    task = {
        "custom_id": arxiv_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": requestify_keyword_extraction(**kwargs)
    }
    return task


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
    import pdb ; pdb.set_trace()

    for i in range(num_chunks):
        chunk = papers[i * requests_batch_size:(i + 1) * requests_batch_size]

        # Write the chunk to a temporary file.
        os.makedirs("data/tmp", exist_ok=True)
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
        key_names = f"data/tmp/keyword-input-ids.txt"
        mode = "w" if i == 0 else "a"
        with open(key_names, mode) as f:  # in case of system failure
            f.write(str(i) + ", " + batch_job.id)

        # Poll for job completion
        batch_jobs.append(
            asyncio.create_task(poll_job(client, i, batch_job.id))
        )

    await asyncio.gather(*batch_jobs)


def send_keywords_to_db():
    """
    Reads all files matching the pattern:
        data/tmp/keyword-output-<5-digit-number>.jsonl

    The function verifies that each entry is successful (status code 200, error is null, and keywords exists).
    It then updates the `paper` table so that for each paper with matching arxiv_id, the `keywords` column is set.
    Returns True if every line was successful and the update completes, otherwise returns False.

    If no output files are found, the function attempts to read the batch job identifiers from the key file
    (data/tmp/keyword-input-ids.txt) and calls poll_job(client, batch_idx, batch_id) to generate the output files.
    """

    file_pattern = "data/tmp/keyword-output-*.jsonl"
    files = glob.glob(file_pattern)

    # If no output files are found, attempt to poll the jobs using the key file.
    if not files:
        key_file = "data/tmp/keyword-input-ids.txt"
        if not os.path.exists(key_file):
            print("Key file not found. No keyword output files available.")
            return False

        try:
            with open(key_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading key file {key_file}: {e}")
            return False

        if not lines:
            print("Key file is empty. No keyword output files available.")
            return False

        # Initialize the client.
        client = OpenAI()
        tasks = []
        for line in lines:
            parts = line.split(",")
            if len(parts) != 2:
                print(f"Invalid line in key file: {line}")
                continue
            try:
                batch_idx = int(parts[0].strip())
                batch_id = parts[1].strip()
            except Exception as e:
                print(f"Error parsing line in key file '{line}': {e}")
                continue
            tasks.append(poll_job(client, batch_idx, batch_id))

        if tasks:
            try:
                # Run all poll_job tasks concurrently.
                asyncio.run(asyncio.gather(*tasks))
            except Exception as e:
                print(f"Error polling batch jobs: {e}")
                return False
        else:
            print("No valid batch job tasks could be created from key file.")
            return False

        # After polling, try to locate the output files again.
        files = glob.glob(file_pattern)
        if not files:
            print("No keyword output files found after polling batch jobs.")
            return False

    # Collect tuples of (keywords, arxiv_id) for the database update.
    updates = []
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue  # Skip empty lines.
                    try:
                        output_entry = json.loads(line)
                    except Exception as e:
                        print(f"Error parsing JSON in file {file_path} at line {line_number}: {e}")
                        return False

                    # Check for errors in the entry.
                    if output_entry.get("error") is not None:
                        print(f"Error found in entry in file {file_path} at line {line_number}: {output_entry.get('error')}")
                        return False

                    response = output_entry.get("response")
                    if response is None or response.get("status_code") != 200:
                        print(f"Non-success status code in file {file_path} at line {line_number}")
                        return False

                    try:
                        keywords = response["body"]["choices"][0]["message"]["content"]
                    except (KeyError, IndexError) as e:
                        print(f"Error extracting keywords in file {file_path} at line {line_number}: {e}")
                        return False

                    if not keywords:
                        print(f"Empty keywords in file {file_path} at line {line_number}")
                        return False

                    arxiv_id = output_entry.get("custom_id")
                    if not arxiv_id:
                        print(f"Missing custom_id in file {file_path} at line {line_number}")
                        return False

                    # Append tuple in order for the parameterized query: (keywords, arxiv_id)
                    updates.append((keywords, arxiv_id))
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return False

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


def classify_batch():
    warnings.warn("Efforts have gone into the regular chat completion API (x2 cost). ")
    asyncio.run(extract_keywords_batch(max_requests=10000, requests_batch_size=10000, max_keywords=7))
    send_keywords_to_db()
