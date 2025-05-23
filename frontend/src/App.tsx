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
    level: -1,
    is_leaf: false,
  });
  const [fetched, setFetched] = useState<boolean>(false);
  const [topics, setTopics] = useState<Topic[]>([]);
  const [tree, setTree] = useState<TopicNode[]>([]);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [minDate, setMinDate] = useState<Date | null>(null);
  const [maxDate, setMaxDate] = useState<Date | null>(null);

  const fetchTopicsTreeAndPapers = async () => {
    if (fetched) return;
    try {
      const [resTopics, resPapers] = await Promise.all([
        axios.get<Topic[]>(`${BACKEND}/topics`),
        axios.get<Paper[]>(`${BACKEND}/papers`),
      ]);
      console.log("fetched topics and papers");

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
        <h1>AI Topics Explorer 2025</h1>
        <div className="header-description">
          <p>
            Explore the latest research in artificial intelligence.{" "}
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
            to classify papers into topics. Also,{" "}
            <strong>please note the following disclaimers</strong>:
          </p>
          <p>
            (1) The results below do not disambiguate researchers with the same
            name. As such, many researchers with the same name could be counted
            as one person,{" "}
            <em>
              artificially inflating their number of publications and hence
              their rankings
            </em>
            .
          </p>
          <p>
            (2) Affiliations and links are being continuously updated, sourced
            via Google search + LLM. They are not 100% accurate and can refer to
            the wrong person - especially same-name researchers.
          </p>
          <p>
            (3) The coarse criteria for a researcher to be included is having published
            more than 20+ papers on arXiv in the past 2 years. This criteria
            ignores the quality of the papers (e.g., acceptance to top
            conferences) and disadvantages researchers who publish fewer but
            highly impactful works. While the number of papers is an indicator
            of productivity, it is not a definite measure of a researcher's
            overall impact. This criteria is used due to data availability.
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
          AI Topics Explorer is licensed under the{" "}
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
