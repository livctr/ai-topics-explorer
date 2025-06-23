import React, { useEffect, useState, useMemo } from "react";
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

      setTopics(resTopics.data);
      setTree(buildTree(resTopics.data));

      const parsedPapers = resPapers.data;
      setPapers(parsedPapers);

      if (parsedPapers.length > 0) {
        const times = parsedPapers.map((p) => new Date(p.date).getTime());
        setMinDate(new Date(Math.min(...times)));
        setMaxDate(new Date(Math.max(...times)));
      }

      setFetched(true);
    } catch (err) {
      console.error("Failed to load topics:", err);
    }
  };

  useEffect(() => {
    fetchTopicsTreeAndPapers();
  }, []);

  // derive all selectedTopic IDs including subtopics
  const selectedTopicAndSubtopics = useMemo(() => {
    if (!selectedTopic || selectedTopic.id === -1) return [];
    const result: Topic[] = [];
    // map from id to node for quick lookup
    const map = new Map<number, TopicNode>();
    const buildMap = (nodes: TopicNode[]) => {
      nodes.forEach((n) => {
        map.set(n.id, n);
        if (n.children.length) buildMap(n.children);
      });
    };
    buildMap(tree);
    // depth-first collect
    const collect = (node: TopicNode) => {
      result.push({
        id: node.id,
        name: node.name,
        parent_id: node.parent_id,
        is_leaf: node.is_leaf,
        level: node.level,
      });
      node.children.forEach(collect);
    };
    const rootNode = map.get(selectedTopic.id);
    if (rootNode) collect(rootNode);
    console.log("Selected topic and subtopics:", result);
    return result;
  }, [selectedTopic, tree]);

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
            Explore the latest research in artificial intelligence. <strong>Click on a topic</strong> to explore its most recent papers
            and the researchers pioneering the field. The data is sourced from the amazing{' '}
            <a href="https://www.semanticscholar.org/product/api">Semantic Scholar API</a>, accessed on{date_string}.
            The topic taxonomy on the left is the result of asking{' '}
            <a href="https://platform.openai.com/docs/models/gpt-4.1-mini">gpt-4.1-mini</a>{' '}
            to classify papers into topics. Also, <strong>please note the following disclaimers</strong>:
          </p>
          <p>(1) The results below do not disambiguate researchers with the same name. As such, many researchers with the same name could be counted as one person, <em>artificially inflating their number of publications and hence their rankings</em>.</p>
          <p>(2) Affiliations and links are being continuously updated, sourced via Tavily search + LLM. They are not 100% accurate and can refer to the wrong person - especially same-name researchers.</p>
          <p>(3) The criteria for paper inclusion is being in the top 50 by citations per month, for the past 12 months (total less than 600 papers). The criteria for author inclusion is being an author on one of those papers and having an H index greater than or equal to 5.</p>
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
          selectedTopicAndSubtopics={selectedTopicAndSubtopics}
        />
      </main>

      <footer>
        <p>AI Topics Explorer is licensed under the{' '}
          <a href="https://www.apache.org/licenses/LICENSE-2.0" target="_blank" rel="noopener noreferrer">Apache License 2.0</a>.
        </p>
        <p>Copyright &copy; <span id="year">{YEAR}</span> by{' '}
          <a href="https://livctr.github.io">Victor Li</a>.
        </p>
      </footer>
    </div>
  );
};

export default App;
