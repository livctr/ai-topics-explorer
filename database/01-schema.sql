/*
Schema for managing research topics, papers, and authorship relationships.
*/

-- CREATE DATABASE cs_research_db;
-- \c cs_research_db

CREATE TABLE topic (
    id INT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT, -- markdown
    parent_id INT REFERENCES topic(id) ON DELETE SET NULL,
    level INT CHECK (level BETWEEN 1 AND 4),
    is_leaf BOOLEAN NOT NULL
);

CREATE TABLE researcher (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    link TEXT,
    affiliation TEXT,  -- primary affiliation
    pub_count INT NOT NULL -- for ordering
);

CREATE INDEX researcher_name_idx ON researcher(name);

CREATE TABLE paper (
    arxiv_id TEXT PRIMARY KEY,
    topic_id INT REFERENCES topic(id) ON DELETE SET NULL,
    topic_from_llm TEXT,
    title TEXT NOT NULL,
    abstract TEXT,
    keywords TEXT, -- Added keywords attribute
    date DATE NOT NULL,
    num_authors INT CHECK (num_authors >= 0)
);

CREATE TABLE writes (
    researcher_id INT REFERENCES researcher(id) ON DELETE CASCADE,
    arxiv_id TEXT REFERENCES paper(arxiv_id) ON DELETE CASCADE,
    author_position INT CHECK (author_position >= 1), -- 1-indexed
    PRIMARY KEY (researcher_id, arxiv_id)
);


CREATE TABLE works_in ( -- limit to 5
    topic_id INT REFERENCES topic(id) ON DELETE CASCADE,
    researcher_id INT REFERENCES researcher(id) ON DELETE CASCADE,
    score DECIMAL(5, 2),
    PRIMARY KEY (topic_id, researcher_id)
);
