from typing_extensions import Self

import json

from pydantic import BaseModel, model_validator
from typing import Optional, List
from datetime import date


# ======================
# Data models
# ======================
class ResearcherLink(BaseModel):
    """Data model for storing researcher search results."""
    id: int
    name: str
    homepage: Optional[str] = None
    affiliations: List[str] = []
    search_date: Optional[str] = None

    @model_validator(mode='after')
    def validate_homepage(self) -> Self:
        if self.homepage is not None and not self.homepage.startswith('http'):
            raise ValueError('homepage must be a valid URL')
        return self
    

class Researcher(BaseModel):
    id: int
    name: str
    url: Optional[str] = None
    affiliations: List[str] = []
    homepage: Optional[str] = None
    h_index: Optional[int] = None

    @model_validator(mode='after')
    def validate_h_index(self) -> Self:
        if self.h_index is not None and self.h_index < 0:
            raise ValueError('h_index must be non-negative')
        return self

class Paper(BaseModel):
    id: str
    url: Optional[str] = None
    title: str
    abstract: Optional[str] = None
    citation_count: Optional[int] = 0
    date: str
    topic_id: Optional[int] = None
    researcher_ids: Optional[List[int]] = []

    @model_validator(mode='after')
    def validate_non_negative(self) -> Self:
        if self.citation_count is not None and self.citation_count < 0:
            raise ValueError('citation_count must be non-negative')
        if self.researcher_ids is not None and len(self.researcher_ids) < 0:
            raise ValueError('Number of authors must be non-negative')
        return self


class Topic(BaseModel):
    id: int
    name: str
    parent_id: Optional[int] = None
    level: int
    is_leaf: bool

    @model_validator(mode='after')
    def validate_level(self) -> Self:
        if self.level == 1 and self.parent_id is not None:
            raise ValueError('level 1 topics cannot have a parent')
        if self.level == 2 and self.parent_id is None:
            raise ValueError('level 2 topics must have a parent')
        if self.level not in (1, 2):
            raise ValueError('level must be 1 or 2')
        return self


class WorksIn(BaseModel):
    researcher_id: int
    topic_id: int
    score: float

    @model_validator(mode='after')
    def validate_score(self) -> Self:
        if self.score < 0:
            raise ValueError('score must be non-negative')
        return self


# ======================
# Data models encapsulation
# ======================
class ScholarInfo(BaseModel):
    date: str
    papers: List[Paper] = []
    researchers: List[Researcher] = []
    topics: List[Topic] = []
    works_in: List[WorksIn] = []

    @model_validator(mode='after')
    def validate_date(self) -> Self:
        try:
            date.fromisoformat(self.date)
        except ValueError:
            raise ValueError('date must be in YYYY-MM-DD format')
        return self
    

def merge_link_info_into_scholar_info(scholar_info: ScholarInfo, links: List[ResearcherLink]) -> ScholarInfo:
    """Merge researcher links into the scholar info."""
    researcher_map = {researcher.id: researcher for researcher in scholar_info.researchers if researcher.id is not None}

    for link in links:
        if link.id is None:
            continue
        researcher = researcher_map.get(link.id)
        if researcher:
            researcher.homepage = link.homepage
            researcher.affiliations = link.affiliations or []

    return scholar_info


LINKS_PATH = "output/researcher_links.json"
INFO_PATH = "output/scholar_info.json"

def write_researcher_links(links: List[ResearcherLink], output_path: str = LINKS_PATH) -> None:
    with open(output_path, 'w') as f:
        f.write(json.dumps([link.model_dump() for link in links], indent=4))


def load_researcher_links_from_file(file_path: str = LINKS_PATH) -> List[ResearcherLink]:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return [ResearcherLink.model_validate(link) for link in data]
    except FileNotFoundError:
        return []


def write_scholar_info(scholar_info: ScholarInfo, output_path: str = INFO_PATH) -> None:
    with open(output_path, 'w') as f:
        f.write(scholar_info.model_dump_json(indent=4))


def load_scholar_info_from_file(
    file_path: str = INFO_PATH,
    research_link_path: Optional[str] = LINKS_PATH,
    merge_links: bool = True
) -> ScholarInfo:
    with open(file_path, 'r') as f:
        data = json.load(f)
    scholar_info = ScholarInfo.model_validate(data)
    if merge_links and research_link_path:
        links = load_researcher_links_from_file(research_link_path)
        scholar_info = merge_link_info_into_scholar_info(scholar_info, links)
    return scholar_info
