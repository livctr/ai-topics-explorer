import argparse
import os
import requests

from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from db_utils import return_conn
from core.data_utils import get_chat_completion


search_results_extraction_template = PromptTemplate(
    input_variables=["stringified_search_results"],
    template=(
        'Given: Search results for "<name> computer science researcher"\n'
        'Task: Return the most relevant link (personal, research group, university) and affiliation (university, company, non-profit)\n'
        "- If multiple researchers match, return: `No link | Unknown`\n"
        "Format: `<link> | <affiliation>`\n\n"
        "{stringified_search_results}"
    )
)


def search_author(name: str, llm: ChatOpenAI, max_results: int = 5):
    """
    researcher name => (link, affiliation) w/ google search + ChatGPT

    Parameters:
    - name: name of the researcher
    - max_results: number between 1 and 10 determining how much of the
        first page ChatGPT sees

    Returns:
    - a dictionary with keys 
    """
    assert 1 <= max_results and max_results <= 10
    query = f"{name} computer science researcher"

    url = "https://www.googleapis.com/customsearch/v1"

    params = {
        "key": os.environ["GOOGLE_SEARCH_API_KEY"],
        "cx": os.environ["GOOGLE_SEARCH_CX"],
        "q": query
    }
    response = requests.get(url, params=params)
    results = response.json().get("items", [])
    results = results[:max_results]  # get top `max_results` results

    # Extract and clean snippets using BeautifulSoup
    cleaned_results = []
    for item in results:
        title = item.get("title", "No Title")
        link = item.get("link", "No Link")
        snippet_html = item.get("htmlSnippet", "")  # This contains raw HTML snippets

        # Clean HTML using BeautifulSoup
        soup = BeautifulSoup(snippet_html, "html.parser")
        cleaned_snippet = soup.get_text()  # Removes HTML tags

        cleaned_results.append({"title": title, "link": link, "snippet": cleaned_snippet})

    stringified_search_results = "\n".join(
        f"- {result['title']} ({result['link']}): {result['snippet']}"
        for result in cleaned_results
    )
    prompt = search_results_extraction_template.invoke(
        {"name": name,
        "stringified_search_results": stringified_search_results}
    )

    return get_chat_completion(llm, prompt)


def update_researcher_links(llm: ChatOpenAI, limit: int = 100, max_results: int = 5) -> None:
    """
    Update the affiliation and link fields of researchers.
    Processes researchers in order of decreasing publication count (pub_count),
    up to the specified limit.

    Parameters:
    - llm: LLM used for decoding the Google search query
    - limit: Google limits 100 queries for free per day
    - max_results: just the top `max_results` are considered, between 1 and 10
    """
    conn = return_conn()
    cur = conn.cursor()
    try:
        # Select researchers ordered by publication count (highest first)
        cur.execute("""
            SELECT id, name
            FROM researcher
            WHERE affiliation IS NULL
            ORDER BY pub_count
            DESC LIMIT %s
        """, (limit,))
        researchers = cur.fetchall()

        for researcher in researchers:
            researcher_id, name = researcher
            print(f"Processing researcher: {name} (ID: {researcher_id})")

            try:
                results = search_author(name, llm, max_results=max_results)
                link, affiliation = tuple([x.strip() for x in results['content'].split('|')])
                if affiliation == "Unknown":
                    raise ValueError("Unknown affiliation")
                if link == "No link":
                    raise ValueError("No link")
            except Exception as e:
                link, affiliation = None, "Unknown"

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
    parser = argparse.ArgumentParser(description="Run researcher link extraction with configurable parameters.")
    
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="The LLM model to use.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for the LLM.")
    parser.add_argument("--max_completion_tokens", type=int, default=75, help="Max tokens for the completion.")
    parser.add_argument("--max_results", type=int, default=5, help="Maximum number of search results to process.")
    parser.add_argument("--limit", type=int, default=1, help="Limit on the number of researcher links to update.")

    args = parser.parse_args()

    llm = ChatOpenAI(
        model=args.model,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens
    )

    update_researcher_links(llm, limit=args.limit, max_results=args.max_results)

    # # Get old researchers table
    # old_researchers_query = """
    # SELECT * FROM researcher;
    # """
    # cur.execute(old_researchers_query)
    # old_researchers = cur.fetchall()

    # old_ids = [old_researcher[0] for old_researcher in old_researchers]
    # old_names = [old_researcher[1] for old_researcher in old_researchers]
    # old_homepages = [old_researcher[2] for old_researcher in old_researchers]
    # old_urls = [old_researcher[3] for old_researcher in old_researchers]
    # old_affiliations = [old_researcher[4] for old_researcher in old_researchers]
    # old_urls_working = check_urls_multithreaded(old_urls)

    
    # # Find the URLs that are still working
    # urls_query = """
    # SELECT id, homepage, url, affiliation
    # FROM researcher
    # WHERE id IN (
    #     SELECT unnest(%s::int[])
    # );
    # """
    # cur.execute(urls_query, (list(authors_dict.keys()),))