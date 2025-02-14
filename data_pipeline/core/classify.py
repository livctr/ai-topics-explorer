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

import warnings

import psycopg2
from tqdm import tqdm

from openai import OpenAI



def requestify_keyword_extraction(abstract: str,
                                  model: str = "gpt-4o-mini",
                                  temperature: float = 0.0,
                                  max_tokens: int = 180,
                                  max_keywords: int = 7,
):
    prompt = (
        f"Extract at most {str(max_keywords)} keywords from the arXiv abstract below, comma-separated. "
        "These keywords should indicate what the paper is about (e.g., topic, problem, method, solution).\n\n"
    )

    return {
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


def extract_keyword(abstract: str):
    """Tests `requestify_keyword_extraction."""
    req = requestify_keyword_extraction(abstract)
    client = OpenAI()
    completion = client.chat.completions.create(**req)
    return completion.choices[0].message.content


def fill_in_keywords_in_db(limit: int = 10, sample_freq: int = 200):
    """
    Iterates over a limited number of papers without keywords, extracts keywords using extract_keyword,
    and updates the database with the results.

    Parameters:
        limit (int): The maximum number of papers to process.
    """
    conn = return_conn()
    cur = conn.cursor()
    
    # Fetch papers with missing or empty keywords, limiting the number of papers
    cur.execute("""
        SELECT arxiv_id, abstract 
        FROM paper 
        WHERE keywords IS NULL OR keywords = ''
        ORDER BY date DESC
        LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    print(f"Number of papers needing keyword extraction: {len(rows)} ")

    i = 0
    
    # Use tqdm to show progress as we iterate over the papers
    for arxiv_id, abstract in tqdm(rows, desc="Getting keywords"):
        try:
            # Extract keywords using the provided function
            keywords = extract_keyword(abstract)

            if i % sample_freq == 0:
                print(f"Updating {arxiv_id} with keywords: {keywords}")
            i += 1

            # Update the record with the extracted keywords
            cur.execute("""
                UPDATE paper
                SET keywords = %s
                WHERE arxiv_id = %s
            """, (keywords, arxiv_id))
            conn.commit()
        except Exception as e:
            print(f"Error processing arxiv_id {arxiv_id}: {e}")
            # Rollback if there's an error and continue with the next record
            conn.rollback()

    cur.close()
    conn.close()


# To run the async function:
if __name__ == "__main__":
    fill_in_keywords_in_db(limit=1000)
