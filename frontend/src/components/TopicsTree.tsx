import React, { useEffect, useState } from "react";
import ListElement from "./ListElement";
import { Topic } from "../types/Topic";

// Extend Topic with a children array.
export type TopicNode = Topic & {
  children: TopicNode[];
};

interface TopicsTreeProps {
  topicsTree: TopicNode[];
  selectedTopic: Topic;
  setSelectedTopic: (topic: Topic) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string) => void;
}

const TopicsTree: React.FC<TopicsTreeProps> = ({
  topicsTree,
  selectedTopic,
  setSelectedTopic,
  setLoading,
  setError,
}) => {
  const [expandedTopics, setExpandedTopics] = useState<Set<number>>(new Set());

  // Automatically expand topics with level <= 1 that are expandable.
  useEffect(() => {
    const initialExpanded = new Set<number>();

    const traverse = (nodes: TopicNode[]) => {
      nodes.forEach((node) => {
        if (node.level <= 1 && !node.is_leaf) {
          initialExpanded.add(node.id);
        }
        if (node.children && node.children.length > 0) {
          traverse(node.children);
        }
      });
    };

    traverse(topicsTree);
    setExpandedTopics(initialExpanded);
  }, [topicsTree]);

  // Toggle the expand/collapse state for a topic.
  const toggleExpand = (id: number) => {
    setExpandedTopics((prev) => {
      const newExpanded = new Set(prev);
      if (newExpanded.has(id)) {
        newExpanded.delete(id);
      } else {
        newExpanded.add(id);
      }
      return newExpanded;
    });
  };

  // When a topic is clicked, set it as selected and toggle expansion if applicable.
  const handleTopicClick = (topic: Topic, e: React.MouseEvent) => {
    // Prevent event bubbling from nested clickable elements.
    e.stopPropagation();
    setSelectedTopic(topic);
    if (!topic.is_leaf) {
      toggleExpand(topic.id);
    }
  };

  // Recursively render topics from the tree structure.
  const renderTopics = (nodes: TopicNode[]) => {
    return nodes.map((node) => (
      <ListElement
        key={node.id}
        className={`topic topic-level-${node.level} ${
          node.is_leaf ? "leaf" : "expandable"
        } ${
          !node.is_leaf && expandedTopics.has(node.id)
            ? "expanded"
            : "collapsed"
        }`}
        onClick={(e) => handleTopicClick(node, e)}
      >
        <div className="topic-header">
          {!node.is_leaf && (
            <span className="topic-toggle">
              <span
                className={
                  expandedTopics.has(node.id)
                    ? "arrow expanded"
                    : "arrow collapsed"
                }
              >
                ▸
              </span>
            </span>
          )}
          {node.id === selectedTopic.id ? (
            <span className="topic-name">
              <strong>{node.name}</strong>
            </span>
          ) : (
            <span className="topic-name">{node.name}</span>
          )}
        </div>
        {/* Render children topics if expanded */}
        {!node.is_leaf &&
          expandedTopics.has(node.id) &&
          node.children.length > 0 && (
            <ul className="topic-children">{renderTopics(node.children)}</ul>
          )}
      </ListElement>
    ));
  };

  return (
    <div className="topic-tree">
      <h3 className="topic-title">Topics</h3>
      <ul className="topic-list list-start">{renderTopics(topicsTree)}</ul>
    </div>
  );
};

export default TopicsTree;
