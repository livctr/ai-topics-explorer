import React, { JSX, useState } from "react";
import { Topic, TopicNode } from "../types/Topic"; // Assuming TopicNode includes a 'children' property

export interface Researcher {
  id: string;
  name: string;
<<<<<<< HEAD
  homepage: string;
  url: string;
=======
  url: string;
  homepage: string;
>>>>>>> agentic_classification
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

<<<<<<< HEAD
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
=======
  // Step 3: Sort these researchers by their score in the selected topic (descending).
  const sortedDisplayResearchers = [...researchersWithScoresForTopic].sort(
    (a, b) => b.scoreInSelectedTopic - a.scoreInSelectedTopic
>>>>>>> agentic_classification
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

<<<<<<< HEAD
  // For a given researcher, get all the topics they work in.
  const getResearcherTopics = (researcherId: string): TopicNode[] => {
    const worksRecords = worksIn.filter(
=======
  const getResearcherTopics = (researcherId: number): TopicNode[] => {
    const researcherWorksInRecords = worksIn.filter(
>>>>>>> agentic_classification
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
<<<<<<< HEAD
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
=======
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
>>>>>>> agentic_classification
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
<<<<<<< HEAD
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
=======
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
>>>>>>> agentic_classification
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