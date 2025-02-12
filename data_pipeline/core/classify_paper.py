from db_utils import return_conn

import os
import json
import tempfile
import psycopg2
import openai
import asyncio


async def extract_keywords(
        max_requests: int = 10,
        requests_b: int = 2,
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

    # Build a list of prompt requests.
    requests = []
    for arxiv_id, abstract in papers:
        prompt = (
            f"Extract at most {str(max_keywords)} keywords from the arXiv abstract below, comma-separated. "
            "These keywords should indicate what the paper is about (e.g., topic, problem, method, solution).\n\n"
            f"{abstract}"
        )
        requests.append({"id": arxiv_id, "prompt": prompt})

    # Partition requests into batches and create a temporary JSONL file for each batch.
    batches = [requests[i:i + requests_b] for i in range(0, len(requests), requests_b)]
    batch_jobs = []  # Each element: {"job_id": ..., "input_filename": ...}
    
    for batch in batches:
        tmp_file = tempfile.NamedTemporaryFile("w+", delete=False, suffix=".jsonl")
        input_filename = tmp_file.name
        for req in batch:
            tmp_file.write(json.dumps(req) + "\n")
        tmp_file.close()

        try:
            # Submit the batch file to the OpenAI Batch API.
            batch_response = openai.Batch.create(
                file=input_filename,
                model="gpt-4o-mini",
                temperature=0.0
            )
            job_id = batch_response["id"]
            batch_jobs.append({"job_id": job_id, "input_filename": input_filename})
        except Exception as e:
            print(f"Failed to submit batch from file {input_filename}: {e}")
            os.remove(input_filename)

    # Create an async lock for database updates.
    db_lock = asyncio.Lock()

    # Helper function to poll for job completion.
    async def poll_job(job_id):
        while True:
            try:
                job_info = openai.Batch.retrieve(job_id)
            except Exception as e:
                print(f"Error retrieving status for job {job_id}: {e}")
                await asyncio.sleep(60)
                continue

            status = job_info.get("status", "")
            if status == "succeeded":
                return job_info.get("output_file")
            elif status in ["failed", "cancelled"]:
                raise Exception(f"Batch job {job_id} {status}.")
            else:
                print(f"Job {job_id} status: {status}. Waiting 60 seconds...")
                await asyncio.sleep(60)

    # For each job, wait for it to finish then process its output file.
    async def process_job(job):
        try:
            output_filename = await poll_job(job["job_id"])
            with open(output_filename, "r") as out_file:
                for line in out_file:
                    try:
                        response_data = json.loads(line)
                        arxiv_id = response_data.get("id")
                        keywords_raw = response_data.get("result", "")
                        if keywords_raw:
                            # Process keywords: trim, replace spaces with underscores, and lowercase.
                            keywords_list = [
                                kw.strip().replace(" ", "_").lower()
                                for kw in keywords_raw.split(",") if kw.strip()
                            ]
                            processed_keywords = ", ".join(keywords_list)
                            # Update the DB in a critical section.
                            async with db_lock:
                                cur.execute(
                                    "UPDATE paper SET keywords = %s WHERE arxiv_id = %s",
                                    (processed_keywords, arxiv_id)
                                )
                                conn.commit()
                    except Exception as inner_e:
                        print(f"Skipping a failed response in {output_filename}: {inner_e}")
            # Delete the output file once processed.
            try:
                os.remove(output_filename)
            except Exception as cleanup_e:
                print(f"Error cleaning up output file {output_filename}: {cleanup_e}")
        except Exception as e:
            print(f"Batch job {job['job_id']} failed: {e}")
        finally:
            # Delete the corresponding input file.
            try:
                os.remove(job["input_filename"])
            except Exception as cleanup_e:
                print(f"Error cleaning up input file {job['input_filename']}: {cleanup_e}")

    # Create and run tasks for all batch jobs concurrently.
    tasks = [asyncio.create_task(process_job(job)) for job in batch_jobs]
    await asyncio.gather(*tasks)

    # Close the database connection.
    cur.close()
    conn.close()

# To run the async function:
if __name__ == "__main__":
    asyncio.run(extract_keywords(requests_b=10, ))



# To run the async function:
if __name__ == "__main__":
    # For example, use a batch size of 10
    asyncio.run(extract_keywords(requests_b=10))





# Example usage:
if __name__ == "__main__":
    import json

    with open("./assets/prompt.txt", "r") as f:
        prompt = f.read()

    example_papers = []
    limit = 3
    i = 0
    with open("./data/arxiv-metadata-oai-snapshot-1mo.json", "r") as f:
        for line in f:
            entry_data = json.loads(line)

            arxiv_id = EntryExtractor.extract_id(entry_data)
            title = EntryExtractor.extract_title(entry_data)
            abstract = EntryExtractor.extract_abstract(entry_data)

            example_papers.append((arxiv_id, title, abstract))

            i += 1
            if i >= limit:
                break

    output = classify_papers(example_papers, prompt)
    print("LLM Output:")
    print(output)
