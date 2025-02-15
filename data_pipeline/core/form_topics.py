from collections import Counter

from data_pipeline.core.db_utils import return_conn


def determine_topic_set(new_topic_threshold: int = 3):
    conn = return_conn()
    cur = conn.cursor()
    
    # Fetch papers with non-empty topic_from_llm, ordering by date.
    cur.execute("""
        SELECT arxiv_id, topic_from_llm
        FROM paper 
        WHERE topic_from_llm IS NOT NULL AND topic_from_llm != ''
        ORDER BY date DESC
    """)
    rows = cur.fetchall()

    # Read the taxonomy from the file.
    with open("assets/taxonomy.txt", 'r', encoding="utf-8") as f:
        template = f.read()
    # Extract taxonomy tags from the text inside the first set of triple backticks.
    taxonomy_set = set(x for x in template.split('```')[1].split('\n') if x != '')
    
    # Count occurrences of each tag from the papers.
    tags = [tag for _, tag in rows]
    tag_counts = Counter(tags)

    # Determine which tags will be used:
    topics = set()
    newly_created_topics = {}
    misc_topics = {}

    for tag, count in tag_counts.items():
        if tag in taxonomy_set:
            topics.add(tag)
        else:
            if count >= new_topic_threshold:
                topics.add(tag)
                newly_created_topics[tag] = count
            else:
                misc_topics[tag] = count

    # Print out details.
    missing_taxonomy_tags = taxonomy_set - topics
    print("Missing taxonomy tags (in taxonomy but not in topics):")
    for tag in missing_taxonomy_tags:
        print(f"  {tag}")

    print("\nNewly created topics (not in taxonomy):")
    for tag, count in newly_created_topics.items():
        print(f"  {tag}: {count}")

    print("\nMiscellaneous topics (below threshold):")
    for tag, count in misc_topics.items():
        print(f"  {tag}: {count}")

    # Create the mapping from arxiv_id to its final tag.
    # For each paper, if its tag is in the taxonomy set or its count is high enough, use that tag;
    # otherwise, use "Miscellaneous".
    arxiv_to_tag = {}
    for arxiv_id, tag in rows:
        if tag in taxonomy_set or tag_counts[tag] >= new_topic_threshold:
            arxiv_to_tag[arxiv_id] = tag
        else:
            arxiv_to_tag[arxiv_id] = "Miscellaneous"
    
    return arxiv_to_tag
 

def flush_and_fill_topics_in_db(arxiv_to_tag: dict):
    """
    Given a dictionary mapping arxiv_id to a tag string (which may include colon-separated hierarchy),
    this function flushes the 'topic' table, populates it with a hierarchy built from the unique tags,
    and updates each paper's topic_id accordingly.
    
    In addition, it drops and creates a materialized view 'works_in' with attributes:
      - topic_id: An ancestor topic's id.
      - researcher_id: The researcher who authored a paper.
      - score: The number of papers written by that researcher whose paper topic is a descendant
        (or the topic itself) of the given topic.
    """
    conn = return_conn()
    cur = conn.cursor()
    
    # Flush out the topic table.
    cur.execute("DELETE FROM topic;")
    conn.commit()
    
    # Build the set of unique tags from the dictionary values.
    unique_tags = set(arxiv_to_tag.values())
    
    # Build a tree structure from the unique tags.
    # We'll represent each topic as a tuple of strings, e.g. ("Theory", "Game Theory & Economics")
    nodes = {}
    for tag in unique_tags:
        # Split and trim each part of the topic.
        parts = [part.strip() for part in tag.split(":")]
        # For each level in the hierarchy, build the path and record the node.
        for i in range(len(parts)):
            path = tuple(parts[: i + 1])
            if path not in nodes:
                parent = tuple(parts[:i]) if i > 0 else None
                nodes[path] = {
                    "name": parts[i],
                    "parent": parent,  # Parent path as tuple (or None)
                    "level": i + 1,
                    "children": set()
                }
            # If not the root, add as a child of its parent.
            if i > 0:
                parent = tuple(parts[:i])
                nodes[parent]["children"].add(path)
    
    # Assign sequential IDs to each node.
    id_map = {}  # maps a node path (tuple) to its assigned id
    next_id = 1
    # Sorting first by level then by path for consistency.
    for path, node in sorted(nodes.items(), key=lambda item: (item[1]['level'], item[0])):
        id_map[path] = next_id
        next_id += 1
    
    # Insert each node into the topic table.
    for path, node in nodes.items():
        node_id = id_map[path]
        parent_id = id_map[node["parent"]] if node["parent"] is not None else None
        level = node["level"]
        is_leaf = (len(node["children"]) == 0)
        # For simplicity, description is left empty.
        cur.execute(
            """
            INSERT INTO topic (id, name, description, parent_id, level, is_leaf)
            VALUES (%s, %s, %s, %s, %s, %s);
            """,
            (node_id, node["name"], "", parent_id, level, is_leaf)
        )
    
    conn.commit()
    
    print("Flushed topic table and inserted the following nodes:")
    for path, node in sorted(nodes.items(), key=lambda item: (item[1]['level'], item[0])):
        print(f"ID: {id_map[path]}, Name: {node['name']}, Level: {node['level']}, "
              f"Parent ID: {id_map[node['parent']] if node['parent'] is not None else None}, "
              f"Is Leaf: {len(node['children']) == 0}")
    
    # Now update the paper table. For each paper, determine its corresponding topic_id.
    # The tag from the dictionary is split in the same way to form the path tuple.
    for arxiv_id, tag in arxiv_to_tag.items():
        parts = [part.strip() for part in tag.split(":")]
        path = tuple(parts)
        topic_id = id_map.get(path)
        if topic_id is None:
            # This should not happen if all tags are processed.
            print(f"Warning: tag '{tag}' for paper {arxiv_id} not found in topic hierarchy.")
            continue
        cur.execute(
            """
            UPDATE paper 
            SET topic_id = %s 
            WHERE arxiv_id = %s;
            """,
            (topic_id, arxiv_id)
        )
    
    conn.commit()
    print("Updated paper table with new topic_id values.")

    # Create the materialized view "works_in".
    # The view's purpose is to record, for each topic and researcher,
    # the number of papers written by that researcher whose paper topic is a descendant
    # (or the topic itself) of the given topic.
    #
    # We accomplish this by using a recursive CTE that, for each paper, walks up the topic tree.
    cur.execute("DROP MATERIALIZED VIEW IF EXISTS works_in;")
    cur.execute("""
        CREATE MATERIALIZED VIEW works_in AS
        WITH RECURSIVE topic_hierarchy AS (
            -- Base case: every topic is an ancestor of itself.
            SELECT id AS descendant, id AS ancestor
            FROM topic
            UNION ALL
            -- Recursive case: if topic t is the parent of a topic already in the hierarchy,
            -- then t is an ancestor of that descendant.
            SELECT th.descendant, t.parent_id AS ancestor
            FROM topic_hierarchy th
            JOIN topic t ON th.ancestor = t.id
            WHERE t.parent_id IS NOT NULL
        )
        SELECT 
            th.ancestor AS topic_id, 
            w.researcher_id, 
            COUNT(*) AS score
        FROM writes w
        JOIN paper p ON w.arxiv_id = p.arxiv_id
        JOIN topic_hierarchy th ON p.topic_id = th.descendant
        GROUP BY th.ancestor, w.researcher_id;
    """)
    conn.commit()
    print("Created materialized view 'works_in'.")


def map_papers_to_topics(new_topic_threshold: int = 3) -> None:
    """
    Adds the topics into the database.
    
    Parameters:
        new_topic_threshold (int): if the topic is not found in the original
            taxonomy, a new topic must appear at least this many times to be
            considered into the database.
    
    Returns:
        None
    """
    arxiv_to_tag = determine_topic_set(new_topic_threshold)
    flush_and_fill_topics_in_db(arxiv_to_tag)
