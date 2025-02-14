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




-- cascade deletion of topics when a paper is deleted, create trigger
CREATE OR REPLACE FUNCTION prune_topic_if_orphan()
RETURNS TRIGGER AS $$
DECLARE
    current_topic_id INT;
    paper_count INT;
    child_count INT;
    topic_is_leaf BOOLEAN;
    parent_topic_id INT;
BEGIN
    -- Start with the topic referenced by the deleted paper.
    current_topic_id := OLD.topic_id;
    IF current_topic_id IS NULL THEN
        RETURN OLD;
    END IF;

    -- Loop upward in the topic hierarchy.
    LOOP
        -- Check if any paper still references the current topic.
        SELECT COUNT(*) INTO paper_count FROM paper WHERE topic_id = current_topic_id;
        IF paper_count > 0 THEN
            -- The topic is still in use.
            EXIT;
        END IF;

        -- Get the current topic's info.
        SELECT is_leaf, parent_id
          INTO topic_is_leaf, parent_topic_id
          FROM topic
         WHERE id = current_topic_id;

        IF NOT FOUND THEN
            -- This topic was already deleted.
            EXIT;
        END IF;

        -- Count the number of child topics.
        SELECT COUNT(*) INTO child_count FROM topic WHERE parent_id = current_topic_id;

        -- If the topic qualifies for deletion:
        IF topic_is_leaf OR child_count = 0 THEN
            DELETE FROM topic WHERE id = current_topic_id;
            -- Propagate upward: check the parent topic.
            current_topic_id := parent_topic_id;
            IF current_topic_id IS NULL THEN
                EXIT;
            END IF;
        ELSE
            -- The topic is not safe to delete (it has children).
            EXIT;
        END IF;
    END LOOP;

    RETURN OLD;
END;
$$ LANGUAGE plpgsql;


CREATE TRIGGER after_paper_delete
AFTER DELETE ON paper
FOR EACH ROW
EXECUTE FUNCTION prune_topic_if_orphan();




-- CREATE TABLE works_in ( -- limit to 5
--     topic_id INT REFERENCES topic(id) ON DELETE CASCADE,
--     researcher_id INT REFERENCES researcher(id) ON DELETE CASCADE,
--     score DECIMAL(5, 2) CHECK (score BETWEEN 0.0 AND 100.0),
--     PRIMARY KEY (topic_id, researcher_id)
-- );
