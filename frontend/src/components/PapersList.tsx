import React from "react";
import { Topic } from "../types/Topic";

export interface Paper {
  arxiv_id: string;
  title: string;
  topic_id: number;
  date: Date;
}

interface PapersListProps {
  selectedTopic: Topic;
  papers: Paper[];
}

const PapersList: React.FC<PapersListProps> = ({
  selectedTopic,
  papers,
}) => {
  // Filter papers that match the selected topic.
  const filteredPapers = papers.filter(
    (paper) => paper.topic_id === selectedTopic.id
  );


  if (filteredPapers.length === 0) {
    if (selectedTopic.is_leaf) {
      return <p>No papers found for {selectedTopic.name}.</p>;
    } else {
      return <p>Click on a sub-topic to see a list of papers!</p>
    }
  }


  return (
    <ul className="papers-list list-start">
      {filteredPapers.map((paper) => (
        <li
          key={paper.arxiv_id}
          style={{
            display: "grid",
            gridTemplateColumns: "auto 1fr",
            alignItems: "start",
          }}
        >
          <span className="topic-toggle">
            <span className="arrow expanded">▸</span>
          </span>
          <a
            href={`https://arxiv.org/abs/${paper.arxiv_id}`}
            target="_blank"
            rel="noopener noreferrer"
          >
            {paper.title} ({paper.date.toLocaleDateString()})
          </a>
        </li>
      ))}
    </ul>
  );
};

export default PapersList;
