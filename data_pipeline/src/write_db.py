import psycopg2
import os
from dotenv import load_dotenv
from src.data_models import ScholarInfo

load_dotenv()


def return_conn():
    conn = psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT")
    )
    return conn


def init_tables_for_ingestion(cur: psycopg2.extensions.cursor):
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS topic (
            id INT PRIMARY KEY,
            name TEXT NOT NULL,
            parent_id INT REFERENCES topic(id) ON DELETE SET NULL,
            level INT CHECK (level BETWEEN 1 AND 2),
            is_leaf BOOLEAN NOT NULL
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS researcher (
            id BIGINT PRIMARY KEY,
            name TEXT NOT NULL,
            homepage TEXT,
            url TEXT,
            affiliation TEXT
        );
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS researcher_name_idx ON researcher(name);
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS paper (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            citation_count INT CHECK (citation_count >= 0),
            url TEXT,
            date DATE NOT NULL,
            topic_id INT REFERENCES topic(id) ON DELETE SET NULL,
            num_authors INT CHECK (num_authors >= 0)
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS works_in (
            researcher_id BIGINT REFERENCES researcher(id) ON DELETE CASCADE,
            topic_id INT REFERENCES topic(id) ON DELETE CASCADE,
            score FLOAT CHECK (score >= 0),
            PRIMARY KEY (researcher_id, topic_id)
        );
    """)


def ingest_scholar_info(cur: psycopg2.extensions.cursor, scholar_info: ScholarInfo):
    # Insert topics
    for topic in scholar_info.topics:
        cur.execute("""
            INSERT INTO topic (id, name, parent_id, level, is_leaf)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING;
        """, (topic.id, topic.name, topic.parent_id, topic.level, topic.is_leaf))

    # Insert researchers
    for researcher in scholar_info.researchers:
        if researcher.h_index > 1:
            cur.execute("""
                INSERT INTO researcher (id, name, homepage, url, affiliation)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING;
            """, (researcher.id, researcher.name, researcher.homepage, researcher.url, researcher.affiliations[0] if researcher.affiliations else None))

    # Insert papers
    for paper in scholar_info.papers:
        cur.execute("""
            INSERT INTO paper (id, title, citation_count, url, date, topic_id, num_authors)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING;
        """, (paper.id, paper.title, paper.citation_count or 0,
              paper.url or None,
              paper.date,
              paper.topic_id,
              len(paper.researcher_ids)))

    # Insert works_in
    for works_in in scholar_info.works_in:
        cur.execute("""
            INSERT INTO works_in (researcher_id, topic_id, score)
            VALUES (%s, %s, %s)
            ON CONFLICT (researcher_id, topic_id) DO NOTHING;
        """, (works_in.researcher_id,
              works_in.topic_id,
              works_in.score))


def update_db(
    scholar_info: ScholarInfo,
):
    conn = return_conn()
    cur = conn.cursor()
    init_tables_for_ingestion(cur)
    ingest_scholar_info(cur, scholar_info)
    conn.commit()
    cur.close()
    conn.close()


if __name__ == "__main__":
    from src.data_models import load_scholar_info_from_file
    scholar_info = load_scholar_info_from_file()
    update_db(scholar_info)
