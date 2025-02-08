import React, { JSX, useState } from "react";
import { Topic } from "../types/Topic";

export interface Researcher {
  id: number;
  name: string;
  link: string;
  affiliation: string;
}

export interface WorksIn {
  topic_id: number;
  researcher_id: number;
  score: number;
}

interface ResearchersListProps {
  topics: Topic[];
  selectedTopicName: string;
  selectedTopicID: number;
  researchers: Researcher[];
  worksIn: WorksIn[];
}

const ResearchersList: React.FC<ResearchersListProps> = ({
  topics,
  selectedTopicName,
  selectedTopicID,
  researchers,
  worksIn,
}) => {
  // Step 1: Filter worksIn records for the selected topic.
  console.log("worksIn: ", worksIn);
  const matchingWorksInRecords = worksIn.filter(
    (record) => record.topic_id === selectedTopicID
  );

  // Step 2: Extract unique researcher IDs from these records.
  const uniqueResearcherIds = Array.from(
    new Set(matchingWorksInRecords.map((record) => record.researcher_id))
  );

  // Step 3: "Join" with the researchers array to get full details.
  const filteredResearchers = uniqueResearcherIds
    .map((id) => researchers.find((researcher) => researcher.id === id))
    .filter((r): r is Researcher => Boolean(r));

  const [expandedResearchers, setExpandedResearchers] = useState<Set<number>>(new Set());

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
  const getResearcherTopics = (researcherId: number): Topic[] => {
    const worksRecords = worksIn.filter((record) => record.researcher_id === researcherId);
    const topicIds = Array.from(new Set(worksRecords.map((record) => record.topic_id)));
    return topics.filter((topic) => topicIds.includes(topic.id));
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
            {t.id === selectedTopicID && index === chain.length - 1 ? (
              <strong>{t.name}</strong>
            ) : (
              t.name
            )}
          </span>
        ))}
      </>
    );
  };

  if (filteredResearchers.length === 0) {
    return (
      <p>
        No researchers found for {selectedTopicName}.
      </p>
    );
  }

  return (
    <ul className="researchers-list list-start">
      {filteredResearchers.map((researcher) => (
        <li key={researcher.id}>
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
          <a href={researcher.link} target="_blank" rel="noopener noreferrer">
            {researcher.name}
          </a>{" "}
          — <em>{researcher.affiliation}</em>
          {expandedResearchers.has(researcher.id) && (
            <div className="researcher-topics" style={{ marginTop: "8px" }}>
              <ul>
                {getResearcherTopics(researcher.id).map((topic) => (
                  <li key={topic.id}>
                    {renderTopicChain(topic)}
                  </li>
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
