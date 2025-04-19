import os
import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from datetime import date, timedelta
from urllib.parse import urlencode
from typing import Tuple, Dict, Any
import time
from tqdm import tqdm
import json
import re

import logging



def remove_unmatched(text: str, open_sym: str, close_sym: str) -> str:
    """
    Remove unmatched occurrences of open_sym and close_sym in text.
    """
    stack = []
    indices_to_remove = set()
    
    # First pass: mark unmatched closing symbols.
    for i, ch in enumerate(text):
        if ch == open_sym:
            stack.append(i)
        elif ch == close_sym:
            if stack:
                stack.pop()
            else:
                indices_to_remove.add(i)
    # Any remaining open symbols in the stack are unmatched.
    indices_to_remove.update(stack)
    
    # Build new text without the unmatched symbols.
    new_text = "".join(ch for i, ch in enumerate(text) if i not in indices_to_remove)
    return new_text

def textify(text: str) -> str:
    # 1. Replace tabs and newlines with spaces and collapse extra spaces.
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    
    # 2. Replace LaTeX commands for italics, bold, sans-serif, and math-sans with their content.
    text = re.sub(r"\\(?:textit|textbf|textsf|mathsf)\{([^}]+)\}", r"\1", text)
    
    # 3. Replace "\%" with "%" and "\times" with " times".
    text = text.replace("\\%", "%").replace("\\times", " times")
    
    # 4. Remove any remaining dollar signs.
    text = text.replace("$", "")
    
    # 5. Remove any \url{...} commands (and their contents).
    text = re.sub(r"\\url\{[^}]*\}", "", text)
    
    # 6. Remove square brackets and whatever is inside them.
    text = re.sub(r'\[[^\]]*\]', '', text)
    
    # 7. Remove unmatched parentheses and curly braces.
    text = remove_unmatched(text, "(", ")")
    text = remove_unmatched(text, "{", "}")
    
    # 8. Detect if the last sentence contains a link and remove it if so.
    #
    # Instead of splitting simply on periods (which can break up URLs that include periods),
    # we split on punctuation followed by whitespace and a capital letter.
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    github_link_pattern = re.compile(r'https://', re.IGNORECASE)

    if sentences:
        last_sentence = sentences[-1].strip()
        if github_link_pattern.search(last_sentence):
            # Remove the last sentence.
            # Find the starting index of the last sentence in the original text.
            idx = text.rfind(last_sentence)
            if idx != -1:
                text = text[:idx].rstrip()
    
    # Ensure the text ends with appropriate punctuation.
    if text and text[-1] not in ".!?":
        text += "."
    
    return text.strip()


class EntryExtractor:
    
    @staticmethod
    def extract_title(entry_data, max_chars: int = 250):
        """Extracts the title from the entry data."""
        title = entry_data.get("title")
        if not title:
            return ""
        title = title.strip()
        title = textify(title)
        if max_chars is not None and len(title) > max_chars:
            front = title[:max_chars // 2]
            back = title[-max_chars // 2:]
            title = front + " ... " + back
        return title

    @staticmethod
    def extract_abstract(entry_data, max_chars: int = 2000):
        """Extracts the abstract from the entry data."""
        abstract = entry_data.get("abstract")
        if not abstract:
            return ""
        abstract = abstract.strip()
        abstract = textify(abstract)
        if max_chars is not None and len(abstract) > max_chars:
            front = abstract[:max_chars // 2]
            back = abstract[-max_chars // 2:]
            abstract = front + " ... " + back
        return abstract


def get_high_relevance_papers_by_date(
    date_filter: str,
    fields_of_study: str = "Computer Science",
    top_k: int = 200,
) -> Dict[str, Any]:

    http = Session()
    http.mount('https://', HTTPAdapter(max_retries=Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"HEAD", "GET", "OPTIONS"}
    )))

    fields = "paperId,title,authors,citationCount,url,abstract,publicationDate"
    sort = "citationCount:desc"

    response = http.get(
        "https://api.semanticscholar.org/graph/v1/paper/search/bulk",
        params={
            'fields': fields + ',openAccessPdf',
            'fieldsOfStudy': fields_of_study,
            'sort': sort,
            'publicationDateOrYear': date_filter,
        }
    )

    response.raise_for_status()  # Ensures we stop if there's an error
    data = response.json()

    if 'data' not in data:
        return {}

    papers = data['data']

    papers_dict = {}
    authors_dict = {}

    for paper in papers[:top_k]:

        if 'paperId' not in paper or paper['paperId'] is None or \
            'authors' not in paper or paper['author'] is None or \
            'publicationDate' not in paper or paper['publicationDate'] is None or \
            'title' not in paper or paper['title'] is None:
            continue

        # Populate authors
        num_authors = 0
        for author in paper['authors']:

            author_id = author.get('authorId') or None
            author_name = author.get('name') or None
            if not author_id or not author_name:
                continue

            num_authors += 1
            if author_id not in authors_dict:
                authors_dict[author_id] = {
                    'name': author_name,
                    'papers': [],
                }
            authors_dict[author_id]['papers'].append(paper['paperId'])

        papers_dict[paper['paperId']] = {
            'title': EntryExtractor.extract_title(paper),
            'abstract': EntryExtractor.extract_abstract(paper),
            'citationCount': paper.get('citationCount') or 0,
            'url': paper.get('url') or '',
            'publicationDate': paper['publicationDate'],
            'numAuthors': num_authors,
        }

    return papers_dict, authors_dict


def get_high_relevance_papers(
    top_per_month: int = 200,
    num_months: int = 12,
    fields_of_study: str = "Computer Science"
) -> Dict[str, Any]:
    """
    Get the top `top_per_month` papers for the past `num_months` in the field of study.
    """
    papers_dict = {}
    authors_dict = {}

    for month in tqdm(range(num_months), desc="Fetching papers by month..."):
        begin_date = date.today() - timedelta(days=30 * (month + 1))
        end_date = date.today() - timedelta(days=30 * month)
        date_filter = f"{begin_date.isoformat()}:{end_date.isoformat()}"

        papers_month, authors_month = \
            get_high_relevance_papers_by_date(date_filter, fields_of_study, top_k=top_per_month)

        papers_dict.update(papers_month)
        for author_id, author_info in authors_month.items():
            name = author_info['name']
            papers = author_info['papers']
            if author_id in authors_dict:
                authors_dict[author_id]['papers'].extend(papers)
            else:
                authors_dict[author_id] = {
                    'name': name,
                    'papers': papers,
                }

            # Likely not necessary
            authors_dict[author_id]['papers'] = list(set(authors_dict[author_id]['papers']))

        # Respect rate limit
        time.sleep(15)

    return papers_dict, authors_dict


def get_author_info(authors_dict: Dict[str, Any], min_paper_cnt: int = 2) -> Dict[str, Any]:
    """
    Get the author information for each author in the authors_dict.
    """
    authors_dict_filtered = {k: v for k, v in authors_dict.items() if len(v['papers']) >= min_paper_cnt}

    author_ids = list(authors_dict_filtered.keys())
    batch_limit = 500

    for i in tqdm(range(0, len(author_ids), batch_limit), desc="Fetching author info..."):
        batch_ids = author_ids[i:i + batch_limit]
        response = requests.post(
            'https://api.semanticscholar.org/graph/v1/author/batch',
            params={'fields': 'authorId,url,name,affiliations,homepage,hIndex'},
            json={"ids": batch_ids}
        )
        response.raise_for_status()
        data = response.json()
        for entry in data:
            author_id = entry.get('authorId') or None
            author_name = entry.get('name') or None
            if not author_id or not author_name:
                continue

            authors_dict_filtered[author_id]['url'] = entry.get('url') or ''
            if author_name != authors_dict_filtered[author_id]['name']:
                logging.debug(f"Name mismatch for author {author_id}: {authors_dict_filtered[author_id]['name']} vs {author_name}")
            authors_dict_filtered[author_id]['name'] = author_name
            authors_dict_filtered[author_id]['affiliations'] = entry.get('affiliations') or []
            authors_dict_filtered[author_id]['homepage'] = entry.get('homepage') or ''
            authors_dict_filtered[author_id]['hIndex'] = entry.get('hIndex') or 0

        # Respect rate limit
        time.sleep(15)

    return authors_dict_filtered


def 


# def get_clustering()


if __name__ == "__main__":

    # papers_dict, authors_dict = get_high_relevance_papers_by_date(
    #     date_filter="2025-03-19:2025-04-19",
    #     fields_of_study="Computer Science"
    # )
    # import pdb ; pdb.set_trace()
    # # Write to json
    # with open('papers.json', 'w') as f:
    #     json.dump(papers_dict, f, indent=4)
    # with open('authors.json', 'w') as f:
    #     json.dump(authors_dict, f, indent=4)

    with open('papers.json', 'r') as f:
        papers_dict = json.load(f)
    with open('authors.json', 'r') as f:
        authors_dict = json.load(f)
    import pdb ; pdb.set_trace()
    authors_dict = get_author_info(authors_dict)
    with open('authors_filtered.json', 'w') as f:
        json.dump(authors_dict, f, indent=4)
    



    


    # print(len(papers))

    # papers = get_high_relevance_papers(num_months=12)
    # import pdb ; pdb.set_trace()
    # # Write to JSON file
    # with open('papers.json', 'w') as f:
    #     import json
    #     json.dump(papers, f, indent=4)

    # with open('papers.json', 'r') as f:
    #     papers = json.load(f)

    # papers = get_high_relevance_papers_2()
    # import pdb ; pdb.set_trace()
    # print(len(papers))
