import React from "react";
import { Topic } from "../types/Topic";

export interface Paper {
  id: string;
  title: string;
  citation_count: number;
  url: string;
  date: string;
  topic_id: number;
}

interface PapersListProps {
  selectedTopics: Topic[];
  papers: Paper[];
}

const PapersList: React.FC<PapersListProps> = ({
  selectedTopics,
  papers,
}) => {
  // Filter papers that is in the List of selected topics.
  const filteredPapers = papers.filter(
    (paper) => selectedTopics.some(
      (topic) => topic.id === paper.topic_id
    )
  );

  if (filteredPapers.length === 0) {
    return <p>No papers found for this topic or its subtopics.</p>
  }

  return (
    <ul className="papers-list list-start">
      {filteredPapers.map((paper) => (
        <li
          key={paper.id}
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
            href={`${paper.url}`}
            target="_blank"
            rel="noopener noreferrer"
          >
            {paper.title} ({paper.date.split('T')[0]})
          </a>
        </li>
      ))}
    </ul>
  );
};

export default PapersList;
