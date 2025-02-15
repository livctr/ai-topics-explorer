import React, { useEffect, useState } from "react";
import TopicsTree from "./components/TopicsTree";
import InfoPanel from "./components/InfoPanel";
import { YEAR, BACKEND } from "./const";
import "./App.css";
import { Topic, TopicNode, buildTree } from "./types/Topic";
import axios from "axios";
import { Paper } from "./components/PapersList";

const App: React.FC = () => {
  const [selectedTopic, setSelectedTopic] = useState<Topic>({
    id: -1,
    name: "",
    parent_id: null,
    is_leaf: false,
    level: -1,
  });
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [fetched, setFetched] = useState<boolean>(false);
  const [topics, setTopics] = useState<Topic[]>([]);
  const [tree, setTree] = useState<TopicNode[]>([]);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [minDate, setMinDate] = useState<Date | null>(null);
  const [maxDate, setMaxDate] = useState<Date | null>(null);

  const fetchTopicsTreeAndPapers = async () => {
    if (fetched) return;
    setLoading(true);
    try {
      const [resTopics, resPapers] = await Promise.all([
        axios.get<Topic[]>(`${BACKEND}/topics`),
        axios.get<Paper[]>(`${BACKEND}/papers`),
      ]);

      // Set topics and tree
      setTopics(resTopics.data);
      setTree(buildTree(resTopics.data));

      // Convert the paper dates to Date objects
      const parsedPapers = resPapers.data.map((p) => ({
        ...p,
        date: new Date(p.date),
      }));
      setPapers(parsedPapers);
      console.log("parsed papers");
      console.log(parsedPapers);

      // Calculate the min and max dates
      if (parsedPapers.length > 0) {
        const times = parsedPapers.map((p) => p.date.getTime());
        const minTime = Math.min(...times);
        const maxTime = Math.max(...times);
        setMinDate(new Date(minTime));
        setMaxDate(new Date(maxTime));
      }

      setFetched(true);
    } catch (err) {
      console.error("Failed to load topics:", err);
      setError("Failed to load topics");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTopicsTreeAndPapers();
  }, []);

  let date_string = "";
  if (minDate && maxDate) {
    date_string = ` and consists of papers from ${minDate.toLocaleDateString()} to ${maxDate.toLocaleDateString()}`;
  }

  return (
    <div className="app-container">
      <header>
        <h1>CS Topics Explorer 2025</h1>
        <div className="header-description">
          <p>
            Explore the latest research in computer science.{" "}
            <strong>Click on a topic</strong> to explore its most recent papers
            and the researchers pioneering the field. The data is sourced from{" "}
            <a href="https://arxiv.org/">arXiv</a> publications made publicly
            available on{" "}
            <a href="https://www.kaggle.com/datasets/Cornell-University/arxiv">
              Kaggle
            </a>
            {date_string}. The topic taxonomy on the left is the result of
            asking{" "}
            <a href="https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/">
              gpt-4o-mini
            </a>{" "}
            to classify papers into topics. Also, please note:
          </p>
          <p>
            (1) The results below do not disambiguate researchers with the same
            name. Sometimes, the affiliation will appear as "Unknown" if that's
            the case.
          </p>
          <p>
            (2) Affiliation and links are being continuously updated, sourced
            via Google search. They may not be 100% accurate.
          </p>
        </div>
      </header>
      <main className="main-content">
        <TopicsTree
          topicsTree={tree}
          selectedTopic={selectedTopic}
          setSelectedTopic={setSelectedTopic}
        />
        <InfoPanel
          topics={topics}
          papers={papers}
          selectedTopic={selectedTopic}
        />
      </main>

      <footer>
        <p>
          CS Topics Explorer is licensed under the{" "}
          <a
            href="https://www.apache.org/licenses/LICENSE-2.0"
            target="_blank"
            rel="noopener noreferrer"
          >
            Apache License 2.0
          </a>
          .
        </p>
        <p>
          Copyright &copy; <span id="year">{YEAR}</span> by{" "}
          <a href="https://livctr.github.io">Victor Li</a>.
        </p>
      </footer>
    </div>
  );
};

export default App;
