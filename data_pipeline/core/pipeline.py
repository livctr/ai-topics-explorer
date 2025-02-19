import argparse
from datetime import datetime
from core.ingest import ingest_arxiv_info
from core.classify import classify_paper_into_topic_with_llm
from core.form_topics import map_papers_to_topics

from db_utils import return_conn


def init_db_if_needed(cur):
    """
    Initialize the database tables and index for topics, researchers, papers, writes, and works_in.
    
    This function creates the tables and index only if they do not already exist.
    
    Parameters:
        conn: A psycopg2 connection object to the PostgreSQL database.
    """
    # Create topic table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS topic (
            id INT PRIMARY KEY,
            name TEXT NOT NULL,
            parent_id INT REFERENCES topic(id) ON DELETE SET NULL,
            level INT CHECK (level BETWEEN 1 AND 4),
            is_leaf BOOLEAN NOT NULL
        );
    """)

    # Create researcher table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS researcher (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            link TEXT,
            affiliation TEXT,  -- primary affiliation
            pub_count INT NOT NULL -- for ordering
        );
    """)

    # Create an index on researcher.name if it doesn't already exist
    cur.execute("""
        CREATE INDEX IF NOT EXISTS researcher_name_idx ON researcher(name);
    """)

    # Create paper table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS paper (
            arxiv_id TEXT PRIMARY KEY,
            topic_id INT REFERENCES topic(id) ON DELETE SET NULL,
            topic_from_llm TEXT,
            title TEXT NOT NULL,
            abstract TEXT,
            keywords TEXT, -- Added keywords attribute
            date DATE NOT NULL,
            num_authors INT CHECK (num_authors >= 0)
        );
    """)

    # Create writes table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS writes (
            researcher_id INT REFERENCES researcher(id) ON DELETE CASCADE,
            arxiv_id TEXT REFERENCES paper(arxiv_id) ON DELETE CASCADE,
            author_position INT CHECK (author_position >= 1), -- 1-indexed
            PRIMARY KEY (researcher_id, arxiv_id)
        );
    """)

    # Create works_in table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS works_in (
            topic_id INT REFERENCES topic(id) ON DELETE CASCADE,
            researcher_id INT REFERENCES researcher(id) ON DELETE CASCADE,
            score DECIMAL(5, 2),
            PRIMARY KEY (topic_id, researcher_id)
        );
    """)


def update_pipeline_state(cur, step: str, completed: bool):
    cur.execute(
        """
        INSERT INTO pipeline_state (step, completed, updated_at)
        VALUES (%s, %s, %s)
        ON CONFLICT (step)
        DO UPDATE SET completed = EXCLUDED.completed, updated_at = EXCLUDED.updated_at
        """,
        (step, completed, datetime.now())
    )

def is_step_completed(cur, step: str) -> bool:
    cur.execute("SELECT completed FROM pipeline_state WHERE step = %s", (step,))
    row = cur.fetchone()
    return row is not None and row[0]


def main():
    parser = argparse.ArgumentParser(
        description="Run the data pipeline."
    )
    # Ingestion parameters
    parser.add_argument(
        "--author_tracking_period_months",
        type=int,
        default=24,
        help="Number of months for tracking author activity."
    )
    parser.add_argument(
        "--author_num_papers_enter_threshold",
        type=int,
        default=7,
        help="Minimum number of papers required for an author to be tracked."
    )
    parser.add_argument(
        "--author_num_papers_keep_threshold",
        type=int,
        default=0,
        help="Minimum number of papers required for an author to be tracked."
    )
    parser.add_argument(
        "--paper_tracking_period_months",
        type=int,
        default=2,
        help="Number of months for tracking paper activity."
    )

    # Classification parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for the LLM (default: 0.0)."
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=40,
        help="Maximum tokens for the LLM's response (default: 40)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of papers to classify per call (default: 1000)."
    )
    parser.add_argument(
        "--sample_freq",
        type=int,
        default=50,
        help="Frequency for sampling papers for classification (default: 50)."
    )
    
    # Topic mapping parameter
    parser.add_argument(
        "--new_topic_threshold",
        type=int,
        default=3,
        help="Threshold for creating a new topic (default: 3)."
    )

    args = parser.parse_args()

    conn = return_conn()
    try:
        with conn.cursor() as cur:

            # Ensure we have tables to fill into
            # init_db_if_needed(cur)
            
            # Ensure pipeline_state table exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_state (
                    step VARCHAR(255) PRIMARY KEY,
                    completed BOOLEAN NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                );
            """)
            conn.commit()

            # Ingestion step
            if not is_step_completed(cur, "ingestion"):
                print("Running ingestion step...")
                ingest_arxiv_info(
                    author_tracking_period_months=args.author_tracking_period_months,
                    author_num_papers_enter_threshold=args.author_num_papers_enter_threshold,
                    author_num_papers_keep_threshold=args.author_num_papers_keep_threshold,
                    paper_tracking_period_months=args.paper_tracking_period_months
                )
                update_pipeline_state(cur, "ingestion", True)
                conn.commit()
            else:
                print("Ingestion step already completed, skipping.")

            # Classification step
            if not is_step_completed(cur, "classification"):
                print("Running classification step...")
                classify_paper_into_topic_with_llm(
                    temperature=args.temperature,
                    max_completion_tokens=args.max_completion_tokens,
                    limit=args.limit,
                    sample_freq=args.sample_freq,
                )
                update_pipeline_state(cur, "classification", True)
                conn.commit()
            else:
                print("Classification step already completed, skipping.")

            # Topic mapping step
            if not is_step_completed(cur, "topic_mapping"):
                print("Running topic mapping step...")
                map_papers_to_topics(
                    new_topic_threshold=args.new_topic_threshold
                )
                update_pipeline_state(cur, "topic_mapping", True)
                conn.commit()
            else:
                print("Topic mapping step already completed, skipping.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
