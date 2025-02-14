import argparse
from datetime import datetime
from core.ingest import ingest_arxiv_info
from core.classify import classify_paper_into_topic_with_llm
from core.form_topics import map_papers_to_topics

from db_utils import return_conn


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
        "--author-tracking-period-months",
        type=int,
        default=24,
        help="Number of months for tracking author activity."
    )
    parser.add_argument(
        "--author-num-papers-enter-threshold",
        type=int,
        default=7,
        help="Minimum number of papers required for an author to be tracked."
    )
    parser.add_argument(
        "--author-num-papers-keep-threshold",
        type=int,
        default=2,
        help="Minimum number of papers required for an author to be tracked."
    )
    parser.add_argument(
        "--paper-tracking-period-months",
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
        "--max-completion-tokens",
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
        "--sample-freq",
        type=int,
        default=50,
        help="Frequency for sampling papers for classification (default: 50)."
    )
    
    # Topic mapping parameter
    parser.add_argument(
        "--new-topic-threshold",
        type=int,
        default=3,
        help="Threshold for creating a new topic (default: 3)."
    )

    args = parser.parse_args()

    conn = return_conn()
    try:
        with conn.cursor() as cur:
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
                    paper_tracking_period_months=args.paper_tracking_periods_months
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
