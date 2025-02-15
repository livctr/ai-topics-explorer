import React, { JSX, useState } from "react";
import { Topic, TopicNode } from "../types/Topic";

export interface Researcher {
  id: number;
  name: string;
  link: string;
  affiliation: string;
  pub_count: number;
}

export interface WorksIn {
  topic_id: number;
  researcher_id: number;
  score: number;
}

interface ResearchersListProps {
  topics: Topic[];
  selectedTopic: Topic;
  researchers: Researcher[];
  worksIn: WorksIn[];
}

const ResearchersList: React.FC<ResearchersListProps> = ({
  topics,
  selectedTopic,
  researchers,
  worksIn,
}) => {
  // Step 1: Filter worksIn records for the selected topic.
  const matchingWorksInRecords = worksIn.filter(
    (record) => record.topic_id === selectedTopic.id
  );

  // Step 2: Extract unique researcher IDs from these records.
  const uniqueResearcherIds = Array.from(
    new Set(matchingWorksInRecords.map((record) => record.researcher_id))
  );

  // Step 3: "Join" with the researchers array to get full details.
  const filteredResearchers = uniqueResearcherIds
    .map((id) => researchers.find((researcher) => researcher.id === id))
    .filter((r): r is Researcher => Boolean(r));

  // Sort researchers by publication count (descending)
  const sortedResearchers = [...filteredResearchers].sort(
    (a, b) => b.pub_count - a.pub_count
  );

  const [expandedResearchers, setExpandedResearchers] = useState<Set<number>>(
    new Set()
  );

  const toggleExpand = (researcherId: number) => {
    setExpandedResearchers((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(researcherId)) {
        newSet.delete(researcherId);
      } else {
        newSet.add(researcherId);
      }
      return newSet;
    });
  };

  // For a given researcher, get all the topics they work in.
  const getResearcherTopics = (researcherId: number): TopicNode[] => {
    const worksRecords = worksIn.filter(
      (record) => record.researcher_id === researcherId
    );
    const topicIds = Array.from(new Set(worksRecords.map((record) => record.topic_id)));
    return topics
            .filter((topic) => topicIds.includes(topic.id))
            .map((topic) => topic as TopicNode);
  };

  // Helper function to render the chain from root to this topic.
  // It uses the parent_id field (root topics have parent_id === null)
  // and renders the chain as "Root > ... > Topic".
  // Only the final topic is bold if its id matches selectedTopicID.
  const renderTopicChain = (topic: Topic): JSX.Element => {
    const chain: Topic[] = [];
    let current: Topic | undefined = topic;
    while (current) {
      // Add current topic to the beginning of the chain
      chain.unshift(current);
      if (current.parent_id === null) break;
      current = topics.find((t) => t.id === current!.parent_id);
    }
    return (
      <>
        {chain.map((t, index) => (
          <span key={t.id}>
            {index > 0 && " > "}
            {t.id === selectedTopic.id ? (
              <strong>{t.name}</strong>
            ) : (
              t.name
            )}
          </span>
        ))}
      </>
    );
  };

  if (sortedResearchers.length === 0) {
    return <p>No researchers found for {selectedTopic.name}.</p>;
  }

  return (
    <ul className="researchers-list list-start">
      {sortedResearchers.map((researcher) => (
        <li
          key={researcher.id}
          style={{
            display: "grid",
            gridTemplateColumns: "auto 1fr",
            alignItems: "start",
          }}
        >
          {/* Triangle toggle */}
          <span
            className="topic-toggle"
            onClick={() => toggleExpand(researcher.id)}
            style={{ cursor: "pointer" }}
          >
            <span
              className={
                expandedResearchers.has(researcher.id)
                  ? "arrow expanded"
                  : "arrow collapsed"
              }
            >
              ▸
            </span>
          </span>
          {/* Researcher details */}
          <div>
            <a href={researcher.link} target="_blank" rel="noopener noreferrer">
              {researcher.name}
            </a>{" "}
            {researcher.affiliation && researcher.affiliation.trim() !== "" && (
              <> — <em>{researcher.affiliation}</em></>
            )}
            {" "}
            <span className="pub-count">
              ({researcher.pub_count})
            </span>
          </div>
          {/* Expanded section spanning both columns */}
          {expandedResearchers.has(researcher.id) && (
            <div style={{ gridColumn: "1 / -1", marginTop: "8px" }}>
              <ul>
                {getResearcherTopics(researcher.id)
                  .filter((topic) => !topic.children || topic.children.length === 0)
                  .map((topic) => (
                    <li key={topic.id}>{renderTopicChain(topic)}</li>
                  ))}
              </ul>
            </div>
          )}
        </li>
      ))}
    </ul>
  );
};

export default ResearchersList;
