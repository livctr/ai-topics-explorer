import psycopg2
import os


def return_conn():
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),  # e.g., "localhost"
        port=os.getenv("DB_PORT")   # e.g., "5432"
    )
    return conn


def execute_query(query):
    conn = return_conn()
    with conn.cursor() as cur:
        cur.execute(query)
        conn.commit()
    conn.close()
