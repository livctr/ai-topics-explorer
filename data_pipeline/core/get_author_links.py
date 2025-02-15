import requests
import psycopg2
from psycopg2 import sql
import os

from db_utils import return_conn

# --- Pretend Google Search API call ---
def search_author(name: str):
    """
    Programmatically search Google using the Google Search API.
    The query is '{name} computer science researcher'.
    Returns a list of tuples: (link, blurb).
    """
    query = f"{name} computer science researcher"
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": os.environ['GOOGLE_SEARCH_API_KEY'],
        "cx": os.environ['GOOGLE_SEARCH_CX'],
        "q": query
    }
    response = requests.get(url, params=params)
    import pdb ; pdb.set_trace()
    results = response.json().get("items", [])

    print(f"Searching for: {query}")

    return results

# --- Pretend affiliation and link extraction ---
def extract_affiliation_and_link(results):
    """
    Given a list of tuples (link, blurb), extract the affiliation and link.
    For this example, we'll simply choose the first result if available.
    Returns (affiliation, link). If nothing is found, returns ("Unknown", None).
    """
    if results:
        # In a real implementation, you'd use more complex logic to parse the blurb.
        # Here, we just split the blurb by commas and assume the first part is the affiliation.
        link, blurb = results[0]
        affiliation = blurb.split(",")[0] if blurb else "Unknown"
        return affiliation, link
    else:
        return "Unknown", None

# --- Update researchers in the database ---
def update_researcher_links(limit: int = 100):
    """
    Update the affiliation and link fields of researchers.
    Processes researchers in order of decreasing publication count (pub_count),
    up to the specified limit.
    """
    conn = return_conn()
    cur = conn.cursor()
    try:
        # Select researchers ordered by publication count (highest first)
        cur.execute("SELECT id, name FROM researcher ORDER BY pub_count DESC LIMIT %s", (limit,))
        researchers = cur.fetchall()

        for researcher in researchers:
            researcher_id, name = researcher
            print(f"Processing researcher: {name} (ID: {researcher_id})")
            # Search for the researcher on Google.
            search_results = search_author(name)
            # Extract affiliation and link.
            affiliation, link = extract_affiliation_and_link(search_results)
            print(f"  Found affiliation: {affiliation} | link: {link}")
            # Update the database.
            cur.execute(
                "UPDATE researcher SET affiliation = %s, link = %s WHERE id = %s",
                (affiliation, link, researcher_id)
            )
            conn.commit()

    except Exception as e:
        print("Error during update:", e)
        conn.rollback()
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    # Update researchers (default limit 100)
    # update_researcher_links(limit=100)
    results = search_author("Carlos Fernandez-Granda")
    import pdb ; pdb.set_trace()

    print(results)
