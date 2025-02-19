import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def return_conn():
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    return conn


def execute_query(query):
    conn = return_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            if cur.description is not None:
                result = cur.fetchall()
            else:
                result = True
            conn.commit()
        return result
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
