from __future__ import annotations
import csv
import os
from datetime import date
import psycopg2
from psycopg2.extensions import connection as _Conn, cursor as _Cur
from typing import Set

from dotenv import load_dotenv

from src.data_models import ScholarInfo, load_scholar_info_from_file

load_dotenv()

DEFAULT_CSV_DIR = "output/csv"

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _none(x):
    """Render None as empty string for CSVs (COPY ... NULL '')"""
    return "" if x is None else x

def _bool(x: bool) -> str:
    return "true" if x else "false"

def _iso_date(x: str) -> str:
    """
    Ensure YYYY-MM-DD. Assumes incoming model already stores ISO strings.
    (If you ever move to datetime objects, adjust here.)
    """
    return x

def _researcher_id_set(scholar: ScholarInfo, min_h: int = 5) -> Set[int]:
    return {r.id for r in scholar.researchers if (r.h_index or 0) >= min_h}

def write_csvs(
    scholar_info: ScholarInfo,
    csv_dir: str = DEFAULT_CSV_DIR,
    min_hindex: int = 5,
) -> None:
    """
    Step 1: Flatten the models into 4 CSVs:
      - topic.csv (id,name,parent_id,level,is_leaf)
      - researcher.csv (id,name,homepage,url,affiliation,h_index)   [filtered h>=min_hindex]
      - paper.csv (id,title,citation_count,url,date,topic_id,num_authors)
      - works_in.csv (researcher_id,topic_id,score)                 [only researcher_id kept above]
    """
    _ensure_dir(csv_dir)
    kept_researchers = _researcher_id_set(scholar_info, min_hindex)

    # 1) topic
    with open(os.path.join(csv_dir, "topic.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "parent_id", "level", "is_leaf"])
        for t in scholar_info.topics:
            w.writerow([t.id, t.name, _none(t.parent_id), t.level, _bool(t.is_leaf)])

    # 2) researcher (filtered)
    with open(os.path.join(csv_dir, "researcher.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "homepage", "url", "affiliation", "h_index"])
        for r in scholar_info.researchers:
            if r.id not in kept_researchers:
                continue
            aff = r.affiliations[0] if getattr(r, "affiliations", None) else None
            w.writerow([r.id, r.name, _none(r.homepage), _none(r.url), _none(aff), (r.h_index or 0)])

    # 3) paper
    with open(os.path.join(csv_dir, "paper.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "title", "citation_count", "url", "date", "topic_id", "num_authors"])
        for p in scholar_info.papers:
            w.writerow([
                p.id,
                p.title,
                p.citation_count or 0,
                _none(p.url),
                _iso_date(p.date),
                _none(p.topic_id),
                len(p.researcher_ids or []),
            ])

    # 4) works_in (filtered by kept researchers)
    with open(os.path.join(csv_dir, "works_in.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["researcher_id", "topic_id", "score"])
        for wi in scholar_info.works_in:
            if wi.researcher_id in kept_researchers:
                w.writerow([wi.researcher_id, wi.topic_id, wi.score])


def return_conn() -> _Conn:
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
    )

def init_tables_for_ingestion(cur: _Cur, clear_tables: bool = True, rebuild_tables: bool = True) -> None:
    if rebuild_tables:
        # Drop in dependency order to avoid FK issues; combine with commas is okay for Postgres
        cur.execute("DROP TABLE IF EXISTS works_in, paper, researcher, topic;")

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
            affiliation TEXT,
            h_index INT CHECK (h_index >= 0)
        );
    """)

    cur.execute("""CREATE INDEX IF NOT EXISTS researcher_name_idx ON researcher(name);""")

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

    if clear_tables:
        cur.execute("TRUNCATE TABLE works_in, paper, researcher, topic;")

def _copy_csv(cur: _Cur, table: str, csv_path: str, columns: str) -> None:
    """
    COPY FROM STDIN using CSV header.
    We set NULL '' so empty strings become NULL for nullable columns.
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        sql = f"""
        COPY {table} ({columns})
        FROM STDIN WITH (FORMAT csv, HEADER true, NULL '', DELIMITER ',', QUOTE '"', ESCAPE '"');
        """
        cur.copy_expert(sql, f)

def ingest_csvs(cur: _Cur, csv_dir: str = DEFAULT_CSV_DIR) -> None:
    """
    Step 2: Read the 4 CSVs and ingest into DB via COPY in FK-safe order:
      topic -> researcher -> paper -> works_in
    """
    # Order matters due to FKs
    _copy_csv(cur, "topic",
              os.path.join(csv_dir, "topic.csv"),
              "id,name,parent_id,level,is_leaf")

    _copy_csv(cur, "researcher",
              os.path.join(csv_dir, "researcher.csv"),
              "id,name,homepage,url,affiliation,h_index")

    _copy_csv(cur, "paper",
              os.path.join(csv_dir, "paper.csv"),
              "id,title,citation_count,url,date,topic_id,num_authors")

    _copy_csv(cur, "works_in",
              os.path.join(csv_dir, "works_in.csv"),
              "researcher_id,topic_id,score")



def run(min_hindex: int = 5, csv_dir: str = DEFAULT_CSV_DIR, clear_tables: bool = True, rebuild_tables: bool = True, copy_csvs: bool = False):
    # Load your existing JSON (and optionally merge links, per your original helper)
    scholar_info = load_scholar_info_from_file()

    # Step 1: materialize to CSVs
    write_csvs(scholar_info, csv_dir=csv_dir, min_hindex=min_hindex)

    # Step 2: create tables and COPY CSVs into Postgres
    if copy_csvs:
        conn = return_conn()
        cur = conn.cursor()
        try:
            init_tables_for_ingestion(cur, clear_tables=clear_tables, rebuild_tables=rebuild_tables)
            ingest_csvs(cur, csv_dir=csv_dir)
            conn.commit()
        finally:
            cur.close()
            conn.close()


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Ingest and process scholar information into Postgres.")
    parser.add_argument("--copy-csvs", action="store_true", help="Copy CSVs into Postgres")
    args = parser.parse_args()
    run(copy_csvs=args.copy_csvs)
