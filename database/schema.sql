/*
Schema for managing research topics, papers, and authorship relationships.
*/

-- CREATE DATABASE cs_research_db;
-- \c cs_research_db

CREATE TABLE topic (
    id INT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    parent_id INT REFERENCES topic(id) ON DELETE SET NULL,
    level INT CHECK (level BETWEEN 1 AND 4),
    is_leaf BOOLEAN NOT NULL
);

CREATE TABLE paper (
    arxiv_id VARCHAR(255) PRIMARY KEY,
    topic_id INT REFERENCES topic(id) ON DELETE SET NULL,
    title VARCHAR(255) NOT NULL,
    num_authors INT CHECK (num_authors >= 0)
);

CREATE TABLE researcher (
    id INT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    link TEXT,
    affiliation VARCHAR(255) NOT NULL  -- primary affiliation
);

CREATE TABLE writes (
    researcher_id INT REFERENCES researcher(id) ON DELETE CASCADE,
    arxiv_id VARCHAR(255) REFERENCES paper(arxiv_id) ON DELETE CASCADE,
    author_position INT CHECK (author_position >= 1), -- 1-indexed
    PRIMARY KEY (researcher_id, arxiv_id)
);

CREATE TABLE works_in (
    topic_id INT REFERENCES topic(id) ON DELETE CASCADE,
    researcher_id INT REFERENCES researcher(id) ON DELETE CASCADE,
    score DECIMAL(5, 2) CHECK (score BETWEEN 0.0 AND 100.0),
    PRIMARY KEY (topic_id, researcher_id)
);
