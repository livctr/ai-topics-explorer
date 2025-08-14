import React, { JSX, useState } from "react";
import { Topic, TopicNode } from "../types/Topic"; // Assuming TopicNode includes a 'children' property

export interface Researcher {
  id: number;
  name: string;
  url: string;
  homepage: string;
  affiliation: string;
  // pub_count: number;
}

export interface WorksIn {
  topic_id: number;
  researcher_id: number;
  score: number;
}

interface ResearchersListProps {
  topics: Topic[]; // Expected to be all topics, used for getResearcherTopics
  selectedTopic: Topic;
  researchers: Researcher[]; // Expected to be all researchers
  worksIn: WorksIn[]; // Expected to be all works_in records
}

interface DisplayResearcher extends Researcher {
  scoreInSelectedTopic: number;
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

  // Step 2: Create a list of researchers with their scores for the selected topic.
  // This list will contain objects of type DisplayResearcher.
  const researchersWithScoresForTopic: DisplayResearcher[] = matchingWorksInRecords
    .map((workRecord) => {
      const researcher = researchers.find(
        (r) => r.id === workRecord.researcher_id
      );
      if (researcher) {
        return {
          ...researcher, // Spread all properties from Researcher
          scoreInSelectedTopic: workRecord.score, // Add the score for the current topic
        };
      }
      return null; // In case a researcher isn't found, though this should be rare if data is consistent
    })
    .filter((r): r is DisplayResearcher => Boolean(r)); // Type guard to filter out nulls

  // Step 3: Sort these researchers by their score in the selected topic (descending).
  const sortedDisplayResearchers = [...researchersWithScoresForTopic].sort(
    (a, b) => b.scoreInSelectedTopic - a.scoreInSelectedTopic
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

  const getResearcherTopics = (researcherId: number): TopicNode[] => {
    const researcherWorksInRecords = worksIn.filter(
      (record) => record.researcher_id === researcherId
    );
    const topicIds = Array.from(
      new Set(researcherWorksInRecords.map((record) => record.topic_id))
    );
    // Assuming `topics` prop contains Topic objects that can be cast to TopicNode
    // and that TopicNode might have a 'children' property used in the filter below.
    return topics
      .filter((topic) => topicIds.includes(topic.id))
      .map((topic) => topic as TopicNode);
  };

  const renderTopicChain = (topic: Topic): JSX.Element => {
    const chain: Topic[] = [];
    let current: Topic | undefined = topic;
    while (current) {
      chain.unshift(current);
      if (current.parent_id === null) break;
      // Find parent in the main topics list
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

  if (sortedDisplayResearchers.length === 0) {
    return <p>No researchers found for {selectedTopic.name}.</p>;
  }

  return (
    <>
      <p>
        <em>
          The number inside the parentheses is the researcher's score for this
          topic (higher means a better match).
        </em>
      </p>
      <ul className="researchers-list list-start">
        {sortedDisplayResearchers.map((displayResearcher) => (
          <li
            key={displayResearcher.id}
            style={{
              display: "grid",
              gridTemplateColumns: "auto 1fr",
              alignItems: "start",
            }}
          >
            <span
              className="topic-toggle"
              onClick={() => toggleExpand(displayResearcher.id)}
              style={{ cursor: "pointer" }}
            >
              <span
                className={
                  expandedResearchers.has(displayResearcher.id)
                    ? "arrow expanded"
                    : "arrow collapsed"
                }
              >
                ▸
              </span>
            </span>
            <div>
              <a
                href={
                  displayResearcher.homepage
                    ? (displayResearcher.homepage.startsWith('http://') || displayResearcher.homepage.startsWith('https://')
                      ? displayResearcher.homepage
                      : `https://${displayResearcher.homepage}`)
                    : (displayResearcher.url.startsWith('http://') || displayResearcher.url.startsWith('https://')
                      ? displayResearcher.url
                      : `https://${displayResearcher.url}`)
                }
                target="_blank"
                rel="noopener noreferrer"
              >
                {displayResearcher.name}
              </a>{" "}
              {displayResearcher.affiliation &&
                displayResearcher.affiliation.trim() !== "" && (
                  <>
                    — <em>{displayResearcher.affiliation}</em>
                  </>
                )}
              {" "}
              <span className="researcher-score">
                ({displayResearcher.scoreInSelectedTopic.toFixed(2)})
              </span>
            </div>
            {expandedResearchers.has(displayResearcher.id) && (
              <div style={{ gridColumn: "1 / -1", marginTop: "8px" }}>
                <ul>
                  {getResearcherTopics(displayResearcher.id)
                    // This filter assumes TopicNode has a 'children' property.
                    // If it's meant to only show leaf topics from the researcher's associations:
                    .filter(
                      (topicNode) => topicNode.is_leaf || (topicNode.children && topicNode.children.length === 0)
                    )
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
