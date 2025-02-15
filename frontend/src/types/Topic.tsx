type Topic = {
  id: number;
  name: string;
  parent_id: number | null;
  is_leaf: boolean;
  level: number;
};

type TopicNode = Topic & {
  children: TopicNode[];
};

const buildTree = (topics: Topic[]): TopicNode[] => {
  const map: { [key: number]: TopicNode } = {};
  const roots: TopicNode[] = [];

  // First, add a children property to every topic and store it in the map.
  topics.forEach((topic) => {
    map[topic.id] = { ...topic, children: [] };
  });

  // Then, build the tree by assigning children to their parents.
  topics.forEach((topic) => {
    const node = map[topic.id];
    if (topic.parent_id === null) {
      // This is a root node.
      roots.push(node);
    } else {
      // Find the parent and add the node to its children.
      const parent = map[topic.parent_id];
      if (parent) {
        parent.children.push(node);
      } else {
        // Optional: handle cases where parent_id is missing in the data.
        console.warn(
          `Parent with id ${topic.parent_id} not found for topic ${topic.id}`
        );
      }
    }
  });

  return roots;
};

export type { Topic, TopicNode };
export { buildTree };
