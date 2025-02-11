from collections import defaultdict
import json
import os
from typing import Callable, List, Dict, Any, Set

from datetime import datetime, timedelta
import regex as re
from tqdm import tqdm

from db_utils import execute_query

import psycopg2

from db_utils import return_conn
from psycopg2.extras import execute_values

SNAPSHOT_PATH = "./data/arxiv-metadata-oai-snapshot.json"
FILTERED_PATH = "./data/arxiv-metadata-oai-snapshot-filtered.json"

def _get_lbl_from_name(names):
    """Tuple (last_name, first_name, middle_name) => String 'first_name [middle_name] last_name'."""
    return [
        name[1] + ' ' + name[0] if name[2] == '' \
        else name[1] + ' ' + name[2] + ' ' + name[0]
        for name in names
    ]


class EntryExtractor:

    @staticmethod
    def extract_id(entry_data):
        """Extracts the arXiv ID from the entry data."""
        return entry_data.get("id")

    @staticmethod
    def extract_submitter(entry_data):
        """Extracts the submitter from the entry data."""
        return entry_data.get("submitter")

    @staticmethod
    def extract_authors(entry_data):
        """
        Extracts and formats author names from the entry data.

        Returns:
            A list of formatted names in the format 'first_name [middle_name] last_name'.
        """
        authors_parsed = entry_data.get("authors_parsed", [])
        return _get_lbl_from_name(authors_parsed)
    
    @staticmethod
    def extract_title(entry_data):
        """Extracts the title from the entry data."""
        return entry_data.get("title").strip()
    
    @staticmethod
    def extract_abstract(entry_data):
        """Extracts the abstract from the entry data."""
        return entry_data.get("abstract").strip()
    
    @staticmethod
    def extract_num_authors(entry_data):
        """Extracts the number of authors from the entry data."""
        authors_parsed = entry_data.get("authors_parsed", [])
        return len(authors_parsed)
    
    @staticmethod
    def extract_categories(entry_data):
        """Extracts the categories from the entry data."""
        return entry_data.get("categories").split(" ")
    
    @staticmethod
    def extract_date(entry_data, first_version: bool = True):
        """Extracts the date from the entry data as a datetime"""

        versions = entry_data.get("versions", [])

        # Select the appropriate version
        version_info = versions[0] if first_version else versions[-1]
        created_str = version_info.get("created")
        if not created_str:
            return False  # No creation date available

        # Parse the date string (e.g., "Sat, 7 Apr 2007 20:23:54 GMT")
        version_date = datetime.strptime(created_str, "%a, %d %b %Y %H:%M:%S %Z")
        return version_date


class PaperFilter:

    @staticmethod
    def is_cs(entry_data):
        """Returns True if the entry is categorized under CS."""
        categories = EntryExtractor.extract_categories(entry_data)
        return any(re.match(r"cs\.[a-zA-Z]{2}", cat) for cat in categories)

    @staticmethod
    def inside_date_range(entry_data, start: datetime, end: datetime, first_version: bool = True):
        """
        Returns True if the paper was submitted between start and end dates.
        Since papers may have multiple versions, `first_version` controls
        whether we consider version 1 (True) or the last version (False).
        """
        try:
            version_date = EntryExtractor.extract_date(entry_data, first_version)
        except IndexError:
            print(f"IndexError on extracting date from {entry_data}")
            return False
        except ValueError:
            print(f"ValueError on extracting date from {entry_data}")
            return False
        # Check if the version date is within the given range
        return start <= version_date <= end


def filter_papers(in_path: str, out_path: str, paper_filters: List[Callable[[Dict[str, Any]], bool]]):
    """
    Filters papers from the input file based on the provided filters and writes the
    filtered papers to the output file.
    """
    with open(in_path, 'r') as f1, open(out_path, 'w') as f2:
        for line in tqdm(f1, desc="Filtering papers..."):
            try:
                entry_data = json.loads(line)
            except json.JSONDecodeError:
                continue  # Skip invalid JSON entries

            # Apply all paper filters (must pass all conditions)
            if all(pf(entry_data) for pf in paper_filters):
                f2.write(line)


def get_researchers_histogram(
    in_path: str,
    paper_filters: List[Callable[[Dict[str, Any]], bool]] = []
):
    """Returns the histogram of researchers and their paper counts."""
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


def update_researcher(
    cur: psycopg2.extensions.cursor,
    researcher_paper_count: Dict[str, int],
    enter_threshold: int = 5,
    keep_threshold: int = 2
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
    for name, count in researcher_paper_count.items():
        if name in existing_names:
            if count >= keep_threshold:
                temp_data.append((name, count))
        else:
            if count >= enter_threshold:
                temp_data.append((name, count))

    # 3. Create a temporary table to hold the "desired" state.
    cur.execute("""
        CREATE TEMPORARY TABLE temp_researchers (
            name VARCHAR(255) PRIMARY KEY,
            pub_count INT
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



def update_paper_and_writes(
    cur: psycopg2.extensions.cursor,
    in_path: str,
) -> Set[str]:
    """
    Updates the paper and writes tables for papers where at least one author is
    an active researcher. For each qualifying paper:
      - Insert or update the paper (with topic_id set to NULL)
      - Insert the corresponding researcher/paper relationships in writes.
    
    Parameters:
      cur: An open psycopg2 cursor (assumed to be within a transaction)
      in_path: The file path to the JSON snapshot of paper data.
    
    Returns:
      A set of arXiv IDs of papers written by active researchers.
    """
    arxiv_ids = set()
    paper_rows = []   # Will accumulate tuples for the paper table
    writes_rows = []  # Will accumulate tuples for the writes table

    # Get the active researchers from the database.
    # This assumes that the researcher table contains only active researchers.
    # We build a mapping from researcher name to researcher id.
    cur.execute("SELECT name, id FROM researcher;")
    active_researchers = {row[0]: row[1] for row in cur.fetchall()}

    # Process the file line by line.
    with open(in_path, 'r') as f:
        for line in tqdm(f, desc="Updating `papers` and `writes`..."):
            try:
                entry_data = json.loads(line)
            except json.JSONDecodeError:
                continue  # Skip invalid JSON entries

            # Extract the list of authors from the entry.
            # get_researchers() should return a list of author names in order.
            researchers_in_paper = EntryExtractor.extract_authors(entry_data)

            # Filter out only the active ones.
            # If none of the paper's authors is an active researcher, skip the paper.
            if not any(name in active_researchers for name in researchers_in_paper):
                continue

            # Extract paper information.
            arxiv_id = EntryExtractor.extract_id(entry_data)
            paper_rows.append((
                arxiv_id,
                None,
                EntryExtractor.extract_title(entry_data),
                EntryExtractor.extract_abstract(entry_data),
                EntryExtractor.extract_date(entry_data, first_version=True).date(),
                EntryExtractor.extract_num_authors(entry_data)  
            ))

            # For each author in the paper, record the relationship if they are active.
            # We assume that get_researchers returns the authors in the proper order.
            for pos, name in enumerate(researchers_in_paper, start=1):
                if name in active_researchers:
                    researcher_id = active_researchers[name]
                    writes_rows.append((researcher_id, arxiv_id, pos))

    # Batch upsert the paper data.
    if paper_rows:
        paper_insert_query = """
            INSERT INTO paper (arxiv_id, topic_id, title, abstract, date, num_authors)
            VALUES %s
            ON CONFLICT (arxiv_id)
            DO UPDATE SET 
                topic_id = EXCLUDED.topic_id,
                title = EXCLUDED.title,
                abstract = EXCLUDED.abstract,
                date = EXCLUDED.date,
                num_authors = EXCLUDED.num_authors;
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






if __name__ == "__main__":
    print(execute_query("SELECT * FROM researcher;"))

    # # Filters for CS papers in the last two years
    # end_date = datetime.today()
    # start_date = end_date - timedelta(days=2*365)
    # last_two_years = lambda x: PaperFilter.inside_date_range(x, start_date, end_date, first_version=True)
    # ids = gather_arxiv_ids(paper_filters=[PaperFilter.is_cs, last_two_years])
    # import pdb ; pdb.set_trace()
    # print("number of ids: ", len(ids))
    # print()

    # from argparse import ArgumentParser

    # parser = ArgumentParser(description="Gather researchers with a minimum paper count.")
    # parser.add_argument("--threshold", type=int, default=5, help="Minimum paper count threshold")
    # parser.add_argument("--start_date", type=str, help="Start date for paper filtering")
    # parser.add_argument("--end_date", type=str, help="End date for paper filtering")
    # args = parser.parse_args()

    # if args.start_date and args.end_date:
    #     start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    #     end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    # elif not args.start_date and not args.end_date:
    #     end_date = datetime.today()
    #     start_date = end_date - timedelta(days=2*365)
    # else:
    #     raise ValueError("Both start_date and end_date must be provided or neither.")

    # is_in_date_range = lambda x: PaperFilter.inside_date_range(x, start_date, end_date, first_version=True)

    # researchers = gather_researchers(
    #     args.threshold,
    #     paper_filters=[PaperFilter.is_cs, is_in_date_range]
    # )
