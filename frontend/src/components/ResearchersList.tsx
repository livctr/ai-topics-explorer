import React, { JSX, useState } from "react";
import { Topic, TopicNode } from "../types/Topic";

export interface Researcher {
  id: string;
  name: string;
  homepage: string;
  url: string;
  affiliation: string;
}

interface ScoredResearcher extends Researcher {
  score: number;
}

export interface WorksIn {
  researcher_id: string;
  topic_id: number;
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

  // Step 4: Build a map from researcher ID → score for this topic.
  const scoreMap = new Map<string, number>(
    matchingWorksInRecords.map(({ researcher_id, score }) => [
      researcher_id,
      score,
    ])
  );

  // Step 5: Sort by score (descending).
  const scoredResearchers: ScoredResearcher[] = filteredResearchers.map(r => ({
    ...r,
    score: scoreMap.get(r.id) ?? 0
  }));

  // Step 6: sort by that score
  const sortedResearchers = scoredResearchers.sort((a, b) => b.score - a.score);

  const [expandedResearchers, setExpandedResearchers] = useState<Set<string>>(
    new Set()
  );

  const toggleExpand = (researcherId: string) => {
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
  const getResearcherTopics = (researcherId: string): TopicNode[] => {
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
    <>
    <p><em>The number inside the parentheses is the number of publications found in roughly the past two years.</em></p>
    <ul className="researchers-list list-start">
      {scoredResearchers.map((researcher) => (
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
            <a href={researcher.url} target="_blank" rel="noopener noreferrer">
              {researcher.name}
            </a>{" "}
            {researcher.affiliation && researcher.affiliation.trim() !== "" && (
              <> — <em>{researcher.affiliation}</em></>
            )}
            {" "}
            <span className="span score">
              ({researcher.score})
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
    </>
  );
};

export default ResearchersList;
