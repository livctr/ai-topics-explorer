from db_utils import return_conn

import os
import json
import tempfile
import psycopg2
import openai
import asyncio
import json


from openai import OpenAI


# Helper function to poll for job completion.
async def poll_job(client, batch_idx, batch_id):
    while True:
        batch_job = client.batches.retrieve(batch_id)

        if batch_job.status == "completed":
            result_file_id = batch_job.output_file_id
            result = client.files.content(result_file_id).content
            result_file_name = f"data/tmp/keyword-output-{str(batch_idx).zfill(5)}.jsonl"
            with open(result_file_name, "wb") as f:
                f.write(result)
        else:
            await asyncio.sleep(60)


def requestify_keyword_extraction(arxiv_id: str,
                                  abstract: str,
                                  model: str = "gpt-4o-mini",
                                  temperature: float = 0.0,
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
            "response_format": { 
                "type": "json_object"
            },
            "messages": [
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": abstract
                }
            ],
        }
    }
    return task


async def extract_keywords(
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

    import pdb ; pdb.set_trace()
    # TODO Ensure papers is correct
    

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
        print(f"Submitting batch {i + 1}/{num_chunks}: {batch_file}")

        # Create batch job
        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        # Poll for job completion
        batch_jobs.append(
            asyncio.create_task(poll_job(client, i, batch_job.id))
        )

    await asyncio.gather(*batch_jobs)


# To run the async function:
if __name__ == "__main__":
    asyncio.run(extract_keywords(max_requests=50, requests_batch_size=5, max_keywords=7))
