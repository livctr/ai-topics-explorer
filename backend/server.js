require("dotenv").config();

const express = require("express");
const { Pool } = require("pg");
const cors = require("cors");

const app = express();
const port = 3000;

const pool = new Pool({
  host: process.env.DB_HOST,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  port: 5432,
});

// Middleware
app.use(cors());
app.use(express.json());

app.get("/", (req, res) => {
  res.json("Healthy");
});


/**
 * GET  
 * Returns all topics. Can specify a given researcher with 
 * `/topics?researcher_id=ID`.
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
      // Fetch only topics associated with the given researcher
      query = `
        SELECT DISTINCT t.id, t.name, t.description, t.parent_id, t.level, t.is_leaf
        FROM topic t
        JOIN works_in w ON t.id = w.topic_id
        WHERE w.researcher_id = $1 AND t.is_leaf = true
        ORDER BY t.id;
      `;
      params.push(researcherID);
    } else {
      // Fetch all topics
      query = "SELECT * FROM topic ORDER BY id;";
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
  function dfs(topic, depth = 1) {
    orderedTopics.push(topic);
    topic.children.forEach((child) => dfs(child, depth + 1));
  }

  rootTopics.forEach((root) => dfs(root));

  return orderedTopics;
}

/**
 * GET /researchers?topic_id=ID
 * Returns a list of researchers working in the given topic, limited to 50.
 */
app.get("/works_in", async (req, res) => {
  try {
    const query = `
      SELECT topic_id, researcher_id, score
      FROM works_in
      ORDER BY topic_id, researcher_id;
    `;
    const result = await pool.query(query);
    res.json(result.rows);
  } catch (error) {
    console.error("Error fetching works_in connections:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

app.get("/researchers", async (req, res) => {
  try {
    const query = `
      SELECT id, name, link, affiliation, pub_count
      FROM researcher
    `;
    const result = await pool.query(query);
    res.json(result.rows);
  } catch (error) {
    console.error("Error fetching researchers:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

app.get("/papers", async (req, res) => {
  try {
    // top 50 papers for each topic
    const query = `
        WITH ranked_papers AS (
          SELECT
            arxiv_id,
            title,
            topic_id,
            date,
            ROW_NUMBER() OVER(PARTITION BY topic_id ORDER BY date DESC) AS rn
          FROM paper
          WHERE topic_id IS NOT NULL
        )
        SELECT
          arxiv_id,
          title,
          topic_id,
          date
        FROM ranked_papers
        WHERE rn <= 50;
    `;
    const result = await pool.query(query);
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
