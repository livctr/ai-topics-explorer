require("dotenv").config();

const express = require("express");
const { Pool } = require("pg");
const cors = require("cors");

const app = express();
const port = 3000;

const pool = new Pool({
  host: process.env.POSTGRES_HOST,
  user: process.env.POSTGRES_USER,
  password: process.env.POSTGRES_PASSWORD,
  database: process.env.POSTGRES_DB,
  port: process.env.POSTGRES_PORT,
});

// Middleware
app.use(cors());
app.use(express.json());

app.get("/", (req, res) => {
  res.json("Healthy");
});

/**
 * GET /topics
 * Returns all topics. Can specify a given researcher with
 * `/topics?researcher_id=ID` to get topics associated with that researcher.
 * When no researcher_id is provided, topics are returned in a hierarchical structure.
 */
app.get("/topics", async (req, res) => {
  try {
    const researcherID = req.query.researcher_id
      ? parseInt(req.query.researcher_id)
      : null;

    // Validate researcher_id if provided
    if (researcherID !== null && isNaN(researcherID)) {
      return res.status(400).json({ error: "Invalid researcher ID" });
    }

    let query;
    let params = [];

    if (researcherID !== null) {
      query = `
        SELECT DISTINCT t.id, t.name, t.parent_id, t.level, t.is_leaf
        FROM topic t
        JOIN works_in w ON t.id = w.topic_id
        WHERE w.researcher_id = $1 AND t.is_leaf = true
        ORDER BY t.id;
      `;
      params.push(researcherID);
    } else {
      query = "SELECT id, name, parent_id, level, is_leaf FROM topic ORDER BY id;";
    }

    const result = await pool.query(query, params);
    const topics = result.rows;

    if (researcherID !== null) {
      return res.json(topics);
    } else {
      // Reorder topics into a hierarchical structure when fetching all topics
      const orderedTopics = orderTopicsHierarchically(topics);
      return res.json(orderedTopics);
    }
  } catch (error) {
    console.error("Error fetching topics:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

/**
 * Converts a flat list of topics into a hierarchical order using DFS.
 * @param {Array} topics - The flat list of topics from the database.
 * @returns {Array} Ordered topics in hierarchical depth-first order.
 */
function orderTopicsHierarchically(topics) {
  const topicMap = new Map();
  const rootTopics = [];

  // Build a map and identify root topics
  topics.forEach((topic) => {
    topic.children = [];
    topicMap.set(topic.id, topic);
    if (topic.parent_id === null) {
      rootTopics.push(topic);
    }
  });

  // Link children to their parents
  topics.forEach((topic) => {
    if (topic.parent_id !== null && topicMap.has(topic.parent_id)) {
      topicMap.get(topic.parent_id).children.push(topic);
    }
  });

  // Perform DFS traversal to get ordered topics
  const orderedTopics = [];
  function dfs(topic) { // depth parameter was unused in its recursive call logic
    orderedTopics.push(topic);
    topic.children.forEach((child) => dfs(child)); // Removed depth
  }

  rootTopics.forEach((root) => dfs(root));

  return orderedTopics;
}

/**
 * GET /researchers?topic_id=ID
 * Returns a list of researchers working in the given topic, limited to 50.
 * The topic_id query parameter is required.
 */
app.get("/researchers", async (req, res) => {
  try {
    const queryText = `
      SELECT r.id, r.name, r.homepage, r.url, r.affiliation, r.h_index
      FROM researcher r
    `;

    const result = await pool.query(queryText);
    res.json(result.rows);
  } catch (error) {
    console.error("Error fetching researchers:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});


/**
 * GET /works_in
 * Returns all connections between researchers and topics from the works_in table.
 */
app.get("/works_in", async (req, res) => {
  try {
    const queryText = `
      SELECT researcher_id, topic_id, score
      FROM works_in
      ORDER BY topic_id, researcher_id;
    `;
    const result = await pool.query(queryText);
    res.json(result.rows);
  } catch (error) {
    console.error("Error fetching works_in connections:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});


app.get("/papers", async (req, res) => {
  try {
    // top 200 papers for each topic by date
    const queryText = `
      WITH ranked_papers AS (
        SELECT
          p.id,
          p.title,
          p.topic_id,
          p.date,
          p.url,
          p.citation_count,
          ROW_NUMBER() OVER(PARTITION BY p.topic_id ORDER BY p.date DESC, p.citation_count DESC NULLS LAST) AS rn
        FROM paper p
        WHERE p.topic_id IS NOT NULL
      )
      SELECT
        id,
        title,
        topic_id,
        date,
        url,
        citation_count
      FROM ranked_papers
      WHERE rn <= 200;
    `;
    const result = await pool.query(queryText);
    res.json(result.rows);
  } catch (error) {
    console.error("Error fetching papers:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

// Start server
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}/`);
});