from collections import defaultdict
import json
import os
import subprocess
from typing import Callable, List, Dict, Any, Set

from datetime import datetime, timedelta
import regex as re
from tqdm import tqdm

import psycopg2

from db_utils import return_conn
from psycopg2.extras import execute_values

from core.data_utils import EntryExtractor, PaperFilter

DATA_DIR = "/app/data/arxiv_data"
SNAPSHOT_PATH = os.path.join(DATA_DIR, "arxiv-metadata-oai-snapshot.json")
FILTERED_PATH = os.path.join(DATA_DIR, "arxiv-metadata-oai-snapshot-filtered.json")


def filter_papers(in_path: str, out_path: str, paper_filters: List[Callable[[Dict[str, Any]], bool]]):
    """
    Filters papers from the input file based on the provided filters and writes the
    filtered papers to the output file.
    """
    i, j = 0, 0
    with open(in_path, 'r') as f1, open(out_path, 'w') as f2:
        for line in tqdm(f1, desc="Filtering papers..."):
            try:
                entry_data = json.loads(line)
            except json.JSONDecodeError:
                continue  # Skip invalid JSON entries

            j += 1
            # Apply all paper filters (must pass all conditions)
            if all(pf(entry_data) for pf in paper_filters):
                f2.write(line)
                i += 1

    print(f"Selected {i} papers out of {j} total papers ({(100*i/j):.2f}%).")
    print(f"Selected papers written to {out_path}")


def get_researchers_histogram(
    in_path: str,
    paper_filters: List[Callable[[Dict[str, Any]], bool]] = [],
):
    """Returns the histogram of researchers and their paper counts.
    
    If a paper has N authors, each author is credited with 1/N.
    """
    researcher_paper_count = defaultdict(int)

    # Read and process the dataset
    with open(in_path, 'r') as f1:
        for line in tqdm(f1, desc="Gathering authors..."):
            try:
                entry_data = json.loads(line)
            except json.JSONDecodeError:
                continue  # Skip invalid JSON entries

            # Apply all paper filters (must pass all conditions)
            if all(pf(entry_data) for pf in paper_filters):
                researchers = EntryExtractor.extract_authors(entry_data)
                for researcher in researchers:
                    researcher_paper_count[researcher] += 1

    return researcher_paper_count



def plot_cumulative_count(freq, save_path=None):
    """
    Plots the cumulative count for a dictionary of {num_words: frequency}.
    
    Parameters:
        freq (dict): Dictionary where keys are numbers (e.g., number of words) and 
                     values are frequencies.
    """
    import matplotlib.pyplot as plt
    import itertools
    # Sort the dictionary keys in increasing order
    sorted_keys = sorted(freq.keys())
    
    # Extract frequencies in the sorted order
    freq_list = [freq[k] for k in sorted_keys]
    
    # Compute cumulative frequencies using itertools.accumulate
    cum_freq = list(itertools.accumulate(freq_list))
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_keys, cum_freq, marker='o', linestyle='-', color='b')
    plt.xlabel("Number of Words")
    plt.ylabel("Cumulative Frequency")
    plt.title("Cumulative Count Plot")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)


def get_title_length_histogram(in_path: str):
    """Returns the histogram of abstract lengths."""
    title_lengths = defaultdict(int)
    title_num_words = defaultdict(int)

    # Read and process the dataset
    with open(in_path, 'r') as f1:
        for line in tqdm(f1, desc="Gathering title lengths..."):
            try:
                entry_data = json.loads(line)
            except json.JSONDecodeError:
                continue  # Skip invalid JSON entries

            title = EntryExtractor.extract_title(entry_data)
            title_lengths[len(title)] += 1
            title_num_words[len(re.findall(r'\w+', title))] += 1

    plot_cumulative_count(title_num_words, save_path="title_num_words.png")
    plot_cumulative_count(title_lengths, save_path="title_lengths.png")
    return title_lengths


def update_researcher(
    cur: psycopg2.extensions.cursor,
    researcher_paper_count: Dict[str, int],
    enter_threshold: int = 7,
    keep_threshold: int = 3,
):
    """
    Batch update the `researcher` table based on the provided
    `researcher_paper_count` dictionary (mapping researcher name to paper count).

    - Existing researchers are updated if their new publication count >= keep_threshold;
      otherwise they are deleted.
    - New researchers are inserted only if their publication count >= enter_threshold.
    """
    # 1. Get all existing researcher names from the table.
    cur.execute("SELECT name FROM researcher;")
    existing_names = {row[0] for row in cur.fetchall()}

    # 2. Prepare the list of researchers to keep/update.
    #    - For existing researchers: keep if count >= keep_threshold.
    #    - For new researchers: insert only if count >= enter_threshold.
    temp_data = []
    for name, i_count in researcher_paper_count.items():
        if name in existing_names:
            if i_count >= keep_threshold:
                temp_data.append((name, i_count))
        else:
            if i_count >= enter_threshold:
                temp_data.append((name, i_count))

    print(f"Number of researchers to consider: {len(temp_data)}")

    # 3. Create a temporary table to hold the "desired" state.
    cur.execute("""
        CREATE TEMPORARY TABLE temp_researchers (
            name VARCHAR(255) PRIMARY KEY,
            pub_count DECIMAL(8, 4)
        ) ON COMMIT DROP;
    """)

    # 4. Bulk insert the temp_data into the temporary table.
    insert_query = "INSERT INTO temp_researchers (name, pub_count) VALUES %s"
    execute_values(cur, insert_query, temp_data)

    # 5. Update existing researchers that are in temp_researchers.
    cur.execute("""
        UPDATE researcher r
        SET pub_count = t.pub_count
        FROM temp_researchers t
        WHERE r.name = t.name;
    """)

    # 6. Insert new researchers from temp_researchers that are not in researcher.
    cur.execute("""
        INSERT INTO researcher (name, pub_count)
        SELECT t.name, t.pub_count
        FROM temp_researchers t
        LEFT JOIN researcher r ON t.name = r.name
        WHERE r.name IS NULL;
    """)

    # 7. Delete any researcher in the main table not present in temp_researchers.
    #    This removes researchers who no longer meet the threshold.
    cur.execute("""
        DELETE FROM researcher
        WHERE name NOT IN (SELECT name FROM temp_researchers);
    """)

    print("Updated `researcher` relation.")


def update_paper_and_writes(
    cur: psycopg2.extensions.cursor,
    in_path: str,
    paper_filters: List[Callable[[Dict[str, Any]], bool]] = [],
) -> Set[str]:
    """
    Updates the paper and writes tables for papers where at least one author is
    an active researcher. For each qualifying paper:
      - Insert or update the paper (with topic_id set to NULL)
      - Insert the corresponding researcher/paper relationships in writes.
    
    Parameters:
      cur: An open psycopg2 cursor (assumed to be within a transaction)
      in_path: The file path to the JSON snapshot of paper data.
      paper_filters: A list of filters to apply to the papers.

    """
    paper_rows = []   # Will accumulate tuples for the paper table
    writes_rows = []  # Will accumulate tuples for the writes table
    seen_arxiv_ids = set()

    # Get the active researchers from the database.
    # This assumes that the researcher table contains only active researchers.
    # We build a mapping from researcher name to researcher id.
    cur.execute("SELECT name, id FROM researcher;")
    active_researchers = {row[0]: row[1] for row in cur.fetchall()}

    # Process the file line by line.
    with open(in_path, 'r') as f:
        for line in tqdm(f, desc="Gathering papers and write relationships..."):
            try:
                entry_data = json.loads(line)
            except json.JSONDecodeError:
                continue  # Skip invalid JSON entries
        
            # if not all
            if not all(pf(entry_data) for pf in paper_filters):
                continue

            # If none of the paper's authors is an active researcher, skip the paper.
            researchers_in_paper = EntryExtractor.extract_authors(entry_data)
            if not any(name in active_researchers for name in researchers_in_paper):
                continue

            # Extract paper information.
            arxiv_id = EntryExtractor.extract_id(entry_data)
            seen_arxiv_ids.add(arxiv_id)
            paper_rows.append((
                arxiv_id,
                None,
                EntryExtractor.extract_title(entry_data, max_chars=245),
                EntryExtractor.extract_abstract(entry_data, max_chars=2000),
                EntryExtractor.extract_date(entry_data, first_version=True).date(),
                EntryExtractor.extract_num_authors(entry_data)  
            ))

            # For each author in the paper, record the relationship if they are active.
            for pos, name in enumerate(researchers_in_paper, start=1):
                if name in active_researchers:
                    researcher_id = active_researchers[name]
                    writes_rows.append((researcher_id, arxiv_id, pos))

    # Batch upsert the paper data.
    print("Number of papers to consider inserting: ", len(paper_rows))
    print("Number of conns to consider inserting: ", len(writes_rows))

    if paper_rows:
        paper_insert_query = """
            INSERT INTO paper (arxiv_id, topic_id, title, abstract, date, num_authors)
            VALUES %s
            ON CONFLICT (arxiv_id)
            DO NOTHING;
        """
        execute_values(cur, paper_insert_query, paper_rows)

    # Batch insert the writes data.
    if writes_rows:
        writes_insert_query = """
            INSERT INTO writes (researcher_id, arxiv_id, author_position)
            VALUES %s
            ON CONFLICT (researcher_id, arxiv_id) DO NOTHING;
        """
        execute_values(cur, writes_insert_query, writes_rows)

    # Delete any papers that are in the database but not in the file.
    # If no papers were seen (i.e., seen_arxiv_ids is empty), delete all papers.
    if seen_arxiv_ids:
        cur.execute(
            "DELETE FROM paper WHERE arxiv_id NOT IN %s;",
            (tuple(seen_arxiv_ids),)
        )
    else:
        cur.execute("DELETE FROM paper;")
    
    print("Updated `paper` and `writes` relations.")

    return seen_arxiv_ids


def ingest_arxiv_info(
         author_tracking_period_months: int = 24,
         author_num_papers_enter_threshold: int = 7,
         author_num_papers_keep_threshold: int = 1,
         paper_tracking_period_months: int = 2,
         redownload: bool = False,
         cleanup_downloaded: bool = True,
         ) -> None:
    """
    Populates the database using the Kaggle metadata dataset.

    Parameters:
    - author_tracking_period_months: length of author track record
    - author_num_papers_enter_threshold: number of publications needed in 
        `author_tracking_period_months` to enter into database
    - author_num_papers_keep_threshold: number of publications need in
        `author_tracking_period_months` to be kept in the database
    - paper_tracking_period_months: includes only recent papers with the first
        version submitted within this many months
    """

    # Track authors who've published the criteria number of papers in the last year
    today = datetime.today()
    author_start_date = today - timedelta(days=author_tracking_period_months*30)

    if redownload:
        # Download the dataset
        print("Downloading dataset...")
        os.makedirs(DATA_DIR, exist_ok=True)
        command = [
            "kaggle", "datasets", "download", "-f", "arxiv-metadata-oai-snapshot.json",
            "-p", DATA_DIR, "--unzip", "Cornell-University/arxiv"
        ]
        subprocess.run(command, check=True)

        filter_papers(
            SNAPSHOT_PATH, 
            FILTERED_PATH,
            [
                PaperFilter.is_ai,  # Only AI papers
                lambda x: PaperFilter.inside_date_range(x, author_start_date, today, first_version=True)
            ]
        )
    else:
        print("Skipping download step.")

    # Get the histogram of researchers and their paper counts
    researcher_paper_count = get_researchers_histogram(FILTERED_PATH)

    # Connect to the database
    conn = return_conn()
    try:
        with conn.cursor() as cur:

            # Perform the updates
            update_researcher(cur,
                                researcher_paper_count,
                                enter_threshold=author_num_papers_enter_threshold,
                                keep_threshold=author_num_papers_keep_threshold)
            del researcher_paper_count

            paper_start_date = today - timedelta(days=paper_tracking_period_months*30)
            update_paper_and_writes(cur, FILTERED_PATH, [
                lambda x: PaperFilter.inside_date_range(
                    x, paper_start_date, today, first_version=True)
            ])

            conn.commit()
    finally:
        conn.close()
    
    if cleanup_downloaded:
        try:
            os.remove(SNAPSHOT_PATH)
        except OSError as e:
            print(f"Error: {e.strerror}")
        
        try:
            os.remove(FILTERED_PATH)
        except OSError as e:
            print(f"Error: {e.strerror}")
