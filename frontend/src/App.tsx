import React, { useEffect, useState } from "react";
import TopicsTree from "./components/TopicsTree";
import InfoPanel from "./components/InfoPanel";
import { YEAR, BACKEND } from "./const";
import "./App.css";
import { Topic, TopicNode, buildTree } from "./types/Topic";
import axios from "axios";





const App: React.FC = () => {
  const [selectedTopic, setSelectedTopic] = useState<[number, string]>([-1, ""]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [fetched, setFetched] = useState<boolean>(false);
  const [topics, setTopics] = useState<Topic[]>([]);
  const [tree, setTree] = useState<TopicNode[]>([]);

  const fetchTopicsTree = async () => {
    if (fetched) return;
    setLoading(true);
    try {
      const response = await axios.get<Topic[]>(`${BACKEND}/topics`);
      const topics = response.data;
      console.log("topics: ", topics);
      setTopics(topics);
      setTree(buildTree(topics));
      setFetched(true);
    } catch (err) {
      console.error("Failed to load topics:", err);
      setError("Failed to load topics");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTopicsTree();
  }, []);

  return (
    <div className="app-container">
      <header>
        <h1>CS Topics Explorer 2025</h1>
        <div className="header-description">
          <p>
            Explore the latest research in computer science.{" "}
            <strong>Click on a topic</strong> to explore what it is, the
            researchers pioneering the field, and its subtopics. The data is
            sourced from the past two years of CS publications on{" "}
            <a href="https://arxiv.org/">arXiv</a> made publicly available on{" "}
            <a href="https://www.kaggle.com/datasets/Cornell-University/arxiv">
              Kaggle
            </a>
            .
          </p>
        </div>
      </header>
      <main className="main-content">
        <TopicsTree
          topicsTree={tree}
          selectedTopic={selectedTopic}
          setSelectedTopic={setSelectedTopic}
          setLoading={setLoading}
          setError={setError}
        />
        <InfoPanel
          topics={topics}
          selectedTopic={selectedTopic}
          loading={loading}
          error={error}
        />
      </main>

      <footer>
        <p>CS Topics Explorer is licensed under the <a href="https://www.apache.org/licenses/LICENSE-2.0" target="_blank" rel="noopener noreferrer">Apache License 2.0</a>.</p>
        <p>Copyright &copy; <span id="year">{YEAR}</span> by <a href="https://livctr.github.io">Victor Li</a>.</p>
      </footer>
    </div>
  );
};

export default App;
