from typing_extensions import Self

import json

from pydantic import BaseModel, model_validator
from typing import Optional, List
from datetime import date


# ======================
# Data models
# ======================

class Researcher(BaseModel):
    id: Optional[int]
    url: Optional[str] = None
    name: str
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


def write_scholar_info(scholar_info: ScholarInfo, output_path: str = "output/scholar_info.json") -> None:
    with open(output_path, 'w') as f:
        f.write(scholar_info.model_dump_json(indent=4))


def load_scholar_info_from_file(file_path: str = "output/scholar_info.json") -> ScholarInfo:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return ScholarInfo.model_validate(data)
