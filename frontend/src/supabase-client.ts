// src/lib/supabaseApi.ts
import { createClient } from "@supabase/supabase-js";
import { Topic, TopicNode } from "./types/Topic";
import { Researcher, WorksIn } from "./components/ResearchersList";
import { Paper } from "./components/PapersList";

// ---- Supabase client ----
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL as string;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY as string;

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

/**
 * Convert a flat list of topics into a DFS-ordered hierarchy, then return a flat
 * list in hierarchical (depth-first) order. Does not mutate the input array.
 */
export function orderTopicsHierarchically(topics: Topic[]): Topic[] {
  // Build a node map with empty children arrays
  const nodeMap = new Map<number, TopicNode>();
  for (const t of topics) {
    nodeMap.set(t.id, { ...t, children: [] });
  }
  console.log("Node map: ", nodeMap);

  // Identify roots and link children to parents
  const roots: TopicNode[] = [];
  for (const t of topics) {
    const node = nodeMap.get(t.id)!;
    if (t.parent_id == null) {
      roots.push(node);
    } else {
      const parent = nodeMap.get(t.parent_id);
      if (parent) parent.children.push(node);
      // If there's a missing parent, we silently treat it as root-like; uncomment to push as root:
      // else roots.push(node);
    }
  }

  // Optional: stabilize child order (by id) for deterministic traversal
  for (const n of nodeMap.values()) {
    n.children.sort((a, b) => a.id - b.id);
  }

  // DFS to produce ordered flat list
  const ordered: Topic[] = [];
  const dfs = (node: TopicNode) => {
    ordered.push(node);                 // TopicNode is a superset of Topic
    node.children.forEach(dfs);
  };
  roots.sort((a, b) => a.id - b.id).forEach(dfs);

  return ordered;
}

// GET /topics  or  /topics?researcher_id=ID
export async function getTopics(researcherId?: number): Promise<Topic[]> {
  if (typeof researcherId === "number") {
    // Join topic with works_in, filter by researcher & leaf
    // Requires FKs so PostgREST can inner join.
    const { data, error } = await supabase
      .from("topic")
      .select("id,name,parent_id,level,is_leaf,works_in!inner(researcher_id)")
      .eq("works_in.researcher_id", researcherId)
      .eq("is_leaf", true)
      .order("id", { ascending: true });
    
    if (error) throw error;

    // Strip the join field
    return (data ?? []).map(({ id, name, parent_id, level, is_leaf }) => ({
      id,
      name,
      parent_id,
      level,
      is_leaf,
    }));
  }

  // All topics (then order hierarchically like server did)
  const { data, error } = await supabase
    .from("topic")
    .select("id,name,parent_id,level,is_leaf")
    .order("id", { ascending: true });

  if (error) throw error;
  return orderTopicsHierarchically((data ?? []) as Topic[]);
}

// GET /researchers
export async function getResearchers(): Promise<Researcher[]> {
  const { data, error } = await supabase
    .from("researcher")
    .select("id,name,homepage,url,affiliation,h_index");

  if (error) throw error;
  return (data ?? []) as Researcher[];
}

// GET /works_in
export async function getWorksIn(): Promise<WorksIn[]> {
  const { data, error } = await supabase
    .from("works_in")
    .select("researcher_id,topic_id,score")
    .order("topic_id", { ascending: true })
    .order("researcher_id", { ascending: true });

  if (error) throw error;
  return (data ?? []) as WorksIn[];
}

// GET /papers  (uses the view)
export async function getPapers(): Promise<Paper[]> {
  const { data, error } = await supabase
    .from("ranked_papers_top200")
    .select("id,title,topic_id,date,url,citation_count");

  if (error) throw error;
  return (data ?? []) as Paper[];
}
