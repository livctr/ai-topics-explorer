import psycopg2
from psycopg2.extras import execute_values
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


def build_topics(
    supercluster_to_clusters,
    super_topics,
    cluster_topics,
    cluster_to_ids,  # cluster_id -> paper_ids
):
    topic_rows = []
    topic_id = 0
    cluster_id_to_topic_id = {}

    for supercluster_id, cluster_ids in supercluster_to_clusters.items():
        # Level 1 topic (supercluster)
        topic_rows.append((
            topic_id,
            super_topics[supercluster_id],
            None,   # parent_id
            1,      # level
            False   # is_leaf
        ))
        parent_id = topic_id
        topic_id += 1

        # Level 2 topics (clusters)
        for cluster_id in cluster_ids:
            topic_rows.append((
                topic_id,
                cluster_topics[cluster_id],
                parent_id,
                2,
                True
            ))
            cluster_id_to_topic_id[cluster_id] = topic_id
            topic_id += 1

    paper_to_topic_id = {}
    for cluster_id, paper_ids in cluster_to_ids.items():
        for paper_id in paper_ids:
            paper_to_topic_id[paper_id] = cluster_id_to_topic_id[cluster_id]

    return topic_rows, paper_to_topic_id


def build_researchers(
    authors_dict
):
    researcher_rows = []
    for author_id, author_info in authors_dict.items():
        homepage = author_info.get("homepage") or None
        url = author_info.get("url") or None

        affiliations = author_info.get("affiliations") or []
        if affiliations is not None and len(affiliations) > 0:
            affiliation = affiliations[0]
        else:
            affiliation = None

        researcher_rows.append((
            author_id,
            author_info["name"],
            homepage,
            url,
            affiliation
        ))
    return researcher_rows


def build_papers(
    papers_dict,
    paper_to_topic_id,  # paper_id -> topic_id
):
    paper_rows = []
    for paper_id, paper_info in papers_dict.items():

        paper_rows.append((
            paper_id,
            paper_info["title"],
            paper_info["citationCount"],
            paper_info["url"],
            paper_info["publicationDate"],
            paper_to_topic_id.get(paper_id, None),
            paper_info["numAuthors"]
        ))
    return paper_rows


def build_works_in(
    authors_dict,  # author_id -> author_info
    papers_dict,  # paper_id -> paper_info
    paper_to_topic_id,  # paper_id -> topic_id  
):
    scores = {}
    for author_id, author_info in authors_dict.items():
        author_papers = author_info.get("papers") or []
        for paper_id in author_papers:
            if paper_id not in paper_to_topic_id or paper_id not in papers_dict:
                continue
            topic_id = paper_to_topic_id[paper_id]
            num_authors = papers_dict[paper_id].get("numAuthors") or 1
            scores[(author_id, topic_id)] = scores.get((author_id, topic_id), 0) + 1.0/num_authors
    works_in_rows = []
    for (author_id, topic_id), score in scores.items():
        works_in_rows.append((author_id, topic_id, score))
    return works_in_rows


def build_data(
    papers_dict,  # paper_id -> paper_info
    authors_dict,  # author_id -> author_info
    supercluster_to_clusters,  # supercluster_id -> cluster_ids
    cluster_to_ids,  # cluster_id -> paper_ids
    super_topics,  # supercluster_id -> topic_name
    cluster_topics,  # cluster_id -> topic_name
):

    topics, paper_to_topic_id = build_topics(
        supercluster_to_clusters,
        super_topics,
        cluster_topics,
        cluster_to_ids
    )
    researchers = build_researchers(authors_dict)
    papers = build_papers(papers_dict, paper_to_topic_id)
    works_in = build_works_in(authors_dict, papers_dict, paper_to_topic_id)
    return topics, researchers, papers, works_in


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
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            homepage TEXT,
            url TEXT,
            affiliation TEXT  -- primary affiliation
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
            researcher_id TEXT REFERENCES researcher(id) ON DELETE CASCADE,
            topic_id INT REFERENCES topic(id) ON DELETE CASCADE,
            score FLOAT CHECK (score >= 0),
            PRIMARY KEY (researcher_id, topic_id)
        );
    """)

def insert_topics(
    cur: psycopg2.extensions.cursor,
    topic_rows,
):
    # Delete old topics
    cur.execute("""DELETE FROM topic;""")
    # Perform batch insert with ON CONFLICT UPDATE
    topic_query = """
    INSERT INTO topic (id, name, parent_id, level, is_leaf)
    VALUES %s
    ON CONFLICT (id)
    DO UPDATE SET
        name = EXCLUDED.name,
        parent_id = EXCLUDED.parent_id,
        level = EXCLUDED.level,
        is_leaf = EXCLUDED.is_leaf;
    """
    execute_values(cur, topic_query, topic_rows)


def update_researchers(
    cur: psycopg2.extensions.cursor,
    researcher_rows,
    delete_old_researchers=True,
):
    # Delete researchers not in the authors_dict
    researcher_ids = [row[0] for row in researcher_rows]
    if delete_old_researchers:
        cur.execute("""
            DELETE FROM researcher
            WHERE id NOT IN (
                SELECT unnest(%s::text[])
            );
        """, (researcher_ids,))
    researcher_query = """
    INSERT INTO researcher (id, name, homepage, url, affiliation)
    VALUES %s
    ON CONFLICT (id)
    DO NOTHING;
    """  # TODO: check homepage/url/affiliation every once in a while
    execute_values(cur, researcher_query, researcher_rows)


def insert_papers(
    cur: psycopg2.extensions.cursor,
    paper_rows,
):
    paper_dates = [paper_row[4] for paper_row in paper_rows]
    paper_dates = [datetime.strptime(date_str, "%Y-%m-%d").date() for date_str in paper_dates]
    paper_rows = [
        tuple(paper_row[:4]) + (date,) + tuple(paper_row[5:])
        for paper_row, date in zip(paper_rows, paper_dates)
    ]

    cur.execute("""DELETE FROM paper;""")
    paper_query = """
    INSERT INTO paper (id, title, citation_count, url, date, topic_id, num_authors)
    VALUES %s
    ON CONFLICT (id)
    DO UPDATE SET
        title = EXCLUDED.title,
        citation_count = EXCLUDED.citation_count,
        url = EXCLUDED.url,
        date = EXCLUDED.date,
        topic_id = EXCLUDED.topic_id,
        num_authors = EXCLUDED.num_authors;
    """
    execute_values(cur, paper_query, paper_rows)


def insert_works_in(
    cur: psycopg2.extensions.cursor,
    works_in_rows,
):
    cur.execute("""DELETE FROM works_in;""")
    works_in_query = """
    INSERT INTO works_in (researcher_id, topic_id, score)
    VALUES %s
    ON CONFLICT (researcher_id, topic_id)
    DO UPDATE SET
        score = EXCLUDED.score;
    """
    execute_values(cur, works_in_query, works_in_rows)


def update_db(data):
    conn = return_conn()
    cur = conn.cursor()
    init_tables_for_ingestion(cur)
    insert_topics(cur, data["topics"])
    update_researchers(cur, data["researchers"])
    insert_papers(cur, data["papers"])
    insert_works_in(cur, data["works_in"])
    conn.commit()
    cur.close()
    conn.close()
