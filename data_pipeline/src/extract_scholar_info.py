"""
This script fetches high-relevance papers and their authors from the Semantic Scholar API.
"""
# Standard library imports
import logging
import re
import time
from datetime import date, timedelta
from typing import Any, Dict, List, Tuple

# Third-party library imports
from requests import Session
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util import Retry

# Local module imports
from src.data_models import Paper, Researcher, ScholarInfo, write_scholar_info

logging.basicConfig(level=logging.INFO)


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
    def extract_title(entry_data: Dict[str, Any], max_chars: int = 250) -> str:
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
    def extract_abstract(entry_data: Dict[str, Any], max_chars: int = 2000) -> str:
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
) -> Tuple[List[Paper], Dict[str, Researcher]]:

    http = Session()
    http.mount('https://', HTTPAdapter(max_retries=Retry(
        total=10,
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

    response.raise_for_status()
    data = response.json()
    papers = data['data']  # Could raise KeyError if 'data' is not present

    parsed_papers: List[Paper] = []
    unique_researchers: Dict[str, Researcher] = {} # Dictionary to store unique researchers by their ID

    for paper_data in papers[:top_k]:
        current_paper_researchers: List[Researcher] = []
        if 'authors' in paper_data and paper_data['authors'] is not None:
            for author_data in paper_data['authors']:
                author_id = author_data.get('authorId')
                if author_id: # Ensure authorId exists to use as a key
                    if author_id not in unique_researchers:
                        # If researcher not seen before, create and store
                        researcher = Researcher(
                            id=author_id,
                            name=author_data.get('name')
                        )
                        unique_researchers[author_id] = researcher
                    # Add the researcher (either newly created or existing) to the current paper's researcher list
                    current_paper_researchers.append(unique_researchers[author_id])

        # Parse paper
        parsed_paper = Paper(
            id=paper_data.get('paperId'),
            url=paper_data.get('url'),
            title=paper_data.get('title'),
            abstract=paper_data.get('abstract'),
            citation_count=paper_data.get('citationCount', 0),
            date=paper_data['publicationDate'],
            researcher_ids=[researcher.id for researcher in current_paper_researchers],
        )
        parsed_papers.append(parsed_paper)
    
    return parsed_papers, unique_researchers


def get_high_relevance_papers(
    top_per_month: int = 100,
    num_months: int = 12,
    fields_of_study: str = "Computer Science"
) -> Tuple[Dict[str, Paper], Dict[str, Researcher]]:
    """
    Get the top `top_per_month` papers for the past `num_months` in the field of study.
    """
    papers_dict = {}
    authors_dict = {}

    for month in tqdm(range(num_months), desc="Fetching papers by month..."):
        begin_date = date.today() - timedelta(days=30 * (month + 1))
        end_date = date.today() - timedelta(days=30 * month)
        date_filter = f"{begin_date.isoformat()}:{end_date.isoformat()}"

        parsed_papers, unique_researchers = get_high_relevance_papers_by_date(
            date_filter,
            fields_of_study,
            top_k=top_per_month
        )

        papers_dict.update({paper.id: paper for paper in parsed_papers})
        authors_dict.update(unique_researchers)

        # Respect rate limit
        time.sleep(15)

    return papers_dict, authors_dict


def fill_author_info(scholar_info: ScholarInfo) -> None:
    """
    Modifies the authors_dict in place with additional information from the Semantic Scholar API.
    """
    authors_dict = {a.id: a for a in scholar_info.researchers if a.h_index is None}
    author_ids = list(authors_dict.keys())
    batch_limit = 1000

    for i in tqdm(range(0, len(author_ids), batch_limit), desc="Fetching author info..."):
        batch_ids = author_ids[i:i + batch_limit]

        http = Session()
        http.mount('https://', HTTPAdapter(max_retries=Retry(
            total=10,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods={"HEAD", "GET", "POST", "OPTIONS"}
        )))

        fields = 'authorId,url,name,affiliations,homepage,hIndex'

        response = http.post(
            'https://api.semanticscholar.org/graph/v1/author/batch',
            params={'fields': fields},
            json={"ids": batch_ids}
        )

        response.raise_for_status()
        authors = response.json()
        # Update researchers
        for researcher_data in authors:
            if researcher_data is None or 'authorId' not in researcher_data:
                continue
            author_id = researcher_data.get('authorId')
            if author_id in authors_dict:
                try:
                    researcher = authors_dict[author_id] if author_id in authors_dict else authors_dict[int(author_id)]
                    researcher.url = researcher_data.get('url', '')
                    researcher.affiliations = researcher_data.get('affiliations', [])
                    researcher.homepage = researcher_data.get('homepage', '')
                    researcher.h_index = researcher_data.get('hIndex', 0)
                except Exception as e:
                    logging.warning(
                        f"Error updating researcher {author_id}: {e}. Data: {researcher_data}"
                    )

        write_scholar_info(scholar_info)

        # Respect rate limit
        time.sleep(15)
