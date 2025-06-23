"""
A lightweight two‑step topic–subtopic classifier for research papers.

Step 1  → Classify the paper into one of the **main topics** listed in
           `assets/cold_start.txt` *or* "Other".
Step 2  → If the paper is **not** "Other", pick / generate a concise sub‑topic.
           The examples provided in `assets/cold_start.txt` act as a cold‑start
           list of candidate sub‑topics.

Compared with the previous agentic workflow this file removes the
state‑graph, AI‑related gate, and memory management logic.  It is a
single, synchronous function that consumes a list of `Paper` objects and
returns the classified papers together with a minimal `Topic` hierarchy.
"""

from __future__ import annotations

# ────────────────────────────── standard library ──────────────────────────────
from copy import deepcopy
from collections import defaultdict
import math
import json
import logging
import os
from typing import Iterable

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# ──────────────────────────────── third‑party ────────────────────────────────
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from typing import Dict, Set, Tuple, Optional, List
# ────────────────────────────────── project ───────────────────────────────────
from src.data_models import Paper, ScholarInfo, Topic, WorksIn, write_scholar_info

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s − %(levelname)s − %(message)s")



class _TopicNode:
    """Node in the TopicsTree representing a single topic."""

    def __init__(self, name: str, parent: Optional["_TopicNode"] = None) -> None:
        self.name = name
        self.parent = parent
        self.children: Dict[str, _TopicNode] = {}
        self.papers: Set[str] = set()

    def add_child(self, name: str) -> "_TopicNode":
        if name not in self.children:
            self.children[name] = _TopicNode(name, parent=self)
        return self.children[name]

    def find_node(self, path: Tuple[str, ...]) -> Optional["_TopicNode"]:
        node = self
        for part in path:
            node = node.children.get(part)
            if node is None:
                return None
        return node

    def copy(self) -> "_TopicNode":
        new_node = _TopicNode(self.name)
        new_node.papers = set(self.papers)
        for child in self.children.values():
            copied = child.copy()
            copied.parent = new_node
            new_node.children[copied.name] = copied
        return new_node


class TopicsTree:
    """Tree of topics, each storing associated paper IDs and classification logic."""

    def __init__(self, other: Optional["TopicsTree"] = None) -> None:
        if other is None:
            self.root = _TopicNode("ROOT")
        else:
            self.root = other.root.copy()

    def load_cold_start(
        self,
        path: str | Path = "assets/cold_start.txt",
        indent_width: int = 4,
    ) -> None:
        """
        Populate the tree from a plain-text seed file such as `assets/cold_start.txt`.

        The file format is:

        • Top-level lines (no leading spaces) → **main topics**  
        • Lines indented by *indent_width* spaces → **sub-topics** of the most
          recent main topic.

        Blank lines are ignored.  Deeper indentation levels are ignored for now
        (they can be added later with another call or by increasing indent_width).

        Args:
            path: Path to the cold-start file.
            indent_width: Number of leading spaces that marks one indentation level.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        current_main: str | None = None
        with path.open(encoding="utf-8") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if not line.strip():        # skip blank / whitespace-only lines
                    continue

                indent = len(line) - len(line.lstrip())
                topic_name = line.strip()

                if indent == 0:
                    # ── Main topic ──
                    current_main = topic_name
                    self.ensure_topic_path((current_main,))
                elif indent == indent_width:
                    # ── Sub-topic ──
                    if current_main is None:
                        raise ValueError(
                            f"Sub-topic '{topic_name}' appears before any main topic"
                        )
                    self.ensure_topic_path((current_main, topic_name))
                else:
                    # For now we silently ignore deeper levels;
                    # extend here if multi-level hierarchies are needed.
                    continue

    def ensure_topic_path(self, topic_path: Tuple[str, ...]) -> None:
        """Ensure that the given topic path exists in the tree, creating nodes as needed."""
        node = self.root
        for part in topic_path:
            node = node.add_child(part)

    def topic_path_exists(self, topic_path: Tuple[str, ...]) -> bool:
        """Return True if the specified topic path exists in the tree."""
        return self.root.find_node(topic_path) is not None

    def add_paper(self, topic_path: Tuple[str, ...], paper_id: str) -> None:
        """Add a paper ID under the given topic path, creating nodes as needed."""
        node = self.root
        for part in topic_path:
            node = node.add_child(part)
        node.papers.add(paper_id)

    def query_subtopics(self, topic_path: Tuple[str, ...]) -> Set[str]:
        """Return the immediate subtopics under the specified path."""
        node = self.root.find_node(topic_path)
        return set(node.children.keys()) if node else set()

    def rename_topic_path(
        self,
        old_path: Tuple[str, ...],
        new_path: Tuple[str, ...]
    ) -> None:
        """Rename a topic node from old_path to new_path, moving its subtree and papers.

        Args:
            old_path: Existing topic path to rename.
            new_path: Desired new topic path.

        Raises:
            KeyError: if old_path does not exist.
            ValueError: if attempting to rename the ROOT node.
        """
        old_node = self.root.find_node(old_path)
        if old_node is None:
            raise KeyError(f"Topic path {old_path} not found")
        if old_node.parent is None:
            raise ValueError("Cannot rename ROOT node")

        # Detach old node from its parent
        old_parent = old_node.parent
        old_name = old_node.name
        del old_parent.children[old_name]

        # Ensure the new parent path exists
        new_parent_path = new_path[:-1]
        new_name = new_path[-1]
        self.ensure_topic_path(new_parent_path)
        new_parent = self.root.find_node(new_parent_path)
        assert new_parent is not None  # should exist now

        # Update node name and parent, then attach
        old_node.name = new_name
        old_node.parent = new_parent
        new_parent.children[new_name] = old_node

    def get_pairings_and_topic_info(
        self
    ) -> Tuple[Dict[str, int], List[Tuple[int, str, Optional[int], int, bool]]]:
        """
        Return:
          paper_id_to_topic_id: Dict[str, int]
            Maps paper IDs to a list of topic IDs they belong to.
          hierarchy:   [(topic_id, topic_name, parent_topic_id, level, is_leaf), ...]
        """
        # Find all nodes that have papers assigned to them or their children.

        # First build this set of nodes
        marked: Set[_TopicNode] = set()
        def mark(node: _TopicNode) -> bool:
            has = bool(node.papers)
            for child in node.children.values():
                if mark(child):
                    has = True
            if has:
                marked.add(node)
            return has

        mark(self.root)

        # Now assign
        paper_id_to_topic_id: Dict[str, int] = {}
        hierarchy: List[Tuple[int, Optional[int], int, bool]] = []
        counter = 1

        def dfs(node: _TopicNode, parent_id: Optional[int], level: int):
            nonlocal counter
            parent_id_ = parent_id if parent_id is None or parent_id > 0 else None  # Make children of root have no parent_id
            for child in node.children.values():
                tid = counter
                child_is_leaf = not any(grandchild in marked for grandchild in child.children.values())
                if child in marked:
                    hierarchy.append((tid, child.name, parent_id_, level + 1, child_is_leaf))

                for paper_id in child.papers:
                    paper_id_to_topic_id[paper_id] = tid  # assumes each paper assigned to only one leaf node

                counter += 1
                dfs(child, tid, level + 1)

        # start from root at level 0; root itself is not assigned an ID
        dfs(self.root, None, 0)
        return paper_id_to_topic_id, hierarchy


def build_classification_choices_in_prompt(
    choices: Iterable[str],
    style: str = "comma",
    add_other_option: bool = False
) -> str:
    """Format an iterable of choices for inclusion in an LLM prompt.

    Args:
        choices: An iterable of choice strings.
        style: One of:
          - "comma":  inline, e.g. "A, B, C"
          - "list":   bulleted, e.g. "- A\n- B\n- C"
        add_other_option: If True, appends "Other" as a choice.

    Returns:
        A single string containing the formatted list.
    """
    # Materialize into a list so we can optionally append
    opts = list(choices)
    if add_other_option:
        opts.append("Other")

    if style == "comma":
        # join with commas and spaces
        return ", ".join(opts)
    elif style == "list":
        # each on its own line prefixed with "- "
        return "\n".join(f"- {opt}" for opt in opts)
    else:
        raise ValueError(f"Unknown style {style!r}; use 'comma' or 'list'.")


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        return levenshtein(b, a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            ins = prev[j + 1] + 1
            dele = curr[j] + 1
            sub = prev[j] + (ca != cb)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def classify_paper(
    paper: Paper,
    topics_tree: TopicsTree,
    llm: ChatOpenAI,
    max_subtopics_per_main_topic: int = 8
) -> None:
    """Classify a paper into the topics tree using an LLM.

    Adds the paper ID under the chosen main topic and subtopic (if any).
    """
    # Build seed topics mapping: main -> list of subtopics
    mains = list(topics_tree.query_subtopics(()))

    # --- Main topic classification ---
    main_choices = build_classification_choices_in_prompt(
        mains, style="comma", add_other_option=False  # Other already an option
    )
    main_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an AI domain expert.  Pick the single best **Topic** from the\n"
            "list for the given paper or output 'Other' if none fit.\n"
            "If the main focus of the paper is to apply AI to some 'non-AI' domain, "
            "classify it as an 'AI Application'.\n\n"
            "Topics: {topics}"),
        HumanMessagePromptTemplate.from_template(
            "Title: {title}\nAbstract: {abstract}")
    ])
    main_chain = main_prompt.partial(topics=main_choices) | llm
    main_resp = main_chain.invoke({
        "title": paper.title,
        "abstract": paper.abstract or "N/A"
    })
    main_topic = main_resp.content.strip()

    if not main_topic in topics_tree.query_subtopics(()):
        main_topic = "Other"

    if main_topic == "Other":
        topics_tree.add_paper((main_topic,), paper.id)
        logging.info(f"Paper {paper.id} ({paper.title}) classified as 'Other'.")
        return

    subtopics = topics_tree.query_subtopics((main_topic,))

    if len(subtopics) < max_subtopics_per_main_topic:
        subtopics_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are an AI domain expert.  Given a paper under the following "
                "**Main Topic**, output a single 2–5-word sub-topic.  "
                "If one of the *sub-topics* matches, reuse it verbatim; "
                "otherwise create a new, equally concise and general sub-topic.\n\n"
                "Main Topic: {main}\n"
                "Sub-topics: {subtopics}"
            ),
            HumanMessagePromptTemplate.from_template(
                "Title: {title}\nAbstract: {abstract}")
        ])
        subtopics_choices = build_classification_choices_in_prompt(
            subtopics, style="comma", add_other_option=False
        )
        subtopics_chain = subtopics_prompt.partial(
            main=main_topic, subtopics=subtopics_choices
        ) | llm
        subtopics_resp = subtopics_chain.invoke({
            "title": paper.title,
            "abstract": paper.abstract or "N/A"
        })
        subtopic = subtopics_resp.content.strip()
        topics_tree.add_paper((main_topic, subtopic), paper.id)
    else:
        # ────────────── reuse or merge into an existing sub-topic ──────────────
        existing_subs: List[str] = sorted(subtopics)   # stable ordering
        options_str = build_classification_choices_in_prompt(
            existing_subs, style="comma", add_other_option=False
        )

        # ── 1) Ask the LLM whether one of the *existing* sub-topics fits ──
        choose_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are an AI domain expert.  Select **one** sub-topic from the "
                "list that best matches the paper *verbatim*.  "
                "If **none** match, respond with\n"
                "    None: <a concise 2-5-word alternative>\n"
                "where the text after the colon is your proposed new label.\n\n"
                "Main Topic: {main}\n"
                "Sub-topics: {options}"
            ),
            HumanMessagePromptTemplate.from_template(
                "Title: {title}\nAbstract: {abstract}")
        ])
        choose_chain = choose_prompt.partial(main=main_topic,
                                             options=options_str) | llm
        raw = choose_chain.invoke({
            "title": paper.title,
            "abstract": paper.abstract or "N/A"
        }).content.strip()

        if raw.lower().startswith("none"):
            # ── 2) The model proposed a *new* label → ask which existing one to merge with ──
            _, _, proposed = raw.partition(":")
            proposed = proposed.strip() or "Miscellaneous"

            merge_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "A new sub-topic **{new_sub}** has been proposed, but we have "
                    "reached the limit of sub-topics under **{main}**.\n"
                    "Select **exactly one** sub-topic from the list that should be "
                    "merged with it, returning the chosen sub-topic *verbatim*.  "
                    "If you believe none are suitable, answer 'None'.\n\n"
                    "Existing Sub-topics: {options}"
                )
            ])
            merge_chain = merge_prompt.partial(
                main=main_topic,
                new_sub=proposed,
                options=options_str
            ) | llm
            merge_choice = merge_chain.invoke({}).content.strip()

            if merge_choice and merge_choice != "None" and merge_choice in existing_subs:
                # Rename the old node to the new proposed label.
                try:
                    topics_tree.rename_topic_path(
                        (main_topic, merge_choice),
                        (main_topic, proposed)
                    )
                except (KeyError, ValueError):
                    # Rename failed (e.g. name collision) → stick with the old label.
                    proposed = merge_choice
            # Record the paper under the chosen (possibly merged) sub-topic.
            topics_tree.add_paper((main_topic, proposed), paper.id)
            subtopic = proposed
        else:
            # LLM picked one of the existing sub-topics directly.
            chosen = raw if raw in existing_subs else existing_subs[0]
            topics_tree.add_paper((main_topic, chosen), paper.id)
            subtopic = chosen
    logging.info(f"Paper {paper.id} ({paper.title}) classified as '{main_topic} / {subtopic}'.")


def build_works_in(
    scholar_info: ScholarInfo,
    alpha: float = 0.6,     # weight on h-index vs. relevance
) -> None:
    # 1) compute raw relevance per (researcher, topic)
    researchers = {r.id for r in scholar_info.researchers}
    topics = {t.id: t for t in scholar_info.topics}
    raw_rel = defaultdict(float)

    for paper in scholar_info.papers:
        if paper.topic_id is None:
            logging.warning(f"Paper ID '{paper.id}' has no topic_id. Skipping.")
            continue

        for r_id in paper.researcher_ids:
            if r_id not in researchers:
                logging.warning(
                    f"Researcher ID '{r_id}' for paper ID '{paper.id}' not found."
                )
                continue

            frac = 1.0 / len(paper.researcher_ids)
            tid = paper.topic_id
            # roll up through the topic hierarchy
            while tid is not None:
                raw_rel[(r_id, tid)] += frac
                parent = topics.get(tid).parent_id if tid in topics else None
                tid = parent

    # turn raw_rel into a flat list so we can scale
    works = [
        WorksIn(researcher_id=r, topic_id=t, score=rel)
        for (r, t), rel in raw_rel.items()
    ]

    if not works:
        scholar_info.works_in = []
        return

    # 2) gather and log-transform h-index for each researcher
    #    we only need those who actually show up in works
    used_rs = {w.researcher_id for w in works}
    h_vals = []
    for r in scholar_info.researchers:
        if r.id in used_rs:
            h_vals.append(math.log1p(r.h_index or 0))

    h_min, h_max = min(h_vals), max(h_vals)

    # build a map r_id → normalized h
    h_norm = {}
    for r in scholar_info.researchers:
        if r.id in used_rs:
            h_log = math.log1p(r.h_index or 0)
            # guard against zero division
            if h_max > h_min:
                h_norm[r.id] = (h_log - h_min) / (h_max - h_min)
            else:
                h_norm[r.id] = 0.0

    # 3) min-max scale raw relevance
    rel_vals = [w.score for w in works]
    rel_min, rel_max = min(rel_vals), max(rel_vals)

    # 4) compute weighted sum and write back to scholar_info
    scholar_info.works_in = []
    for w in works:
        if rel_max > rel_min:
            rel_s = (w.score - rel_min) / (rel_max - rel_min)
        else:
            rel_s = 0.0

        combined = alpha * h_norm[w.researcher_id] + (1 - alpha) * rel_s

        scholar_info.works_in.append(
            WorksIn(
                researcher_id=w.researcher_id,
                topic_id=w.topic_id,
                score=combined
            )
        )


def run_topics_classification(
    scholar_info: ScholarInfo,
) -> None:
    """Classify a list of papers into the topics tree using an LLM."""
    papers = scholar_info.papers
    scholar_info.topics = []

    tree = TopicsTree()
    tree.load_cold_start()

    _llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0,
                  openai_api_key=os.getenv("OPENAI_API_KEY"))

    for paper in papers:
        classify_paper(paper, tree, _llm, max_subtopics_per_main_topic=8)
    
    paper_id_to_topic_id, hierarchy = tree.get_pairings_and_topic_info()

    # Build topics
    scholar_info.topics = [
        Topic(id=tid, name=name, parent_id=parent_id, level=level, is_leaf=is_leaf)
        for tid, name, parent_id, level, is_leaf in hierarchy
    ]

    # Fill topic IDs in papers
    for paper in papers:
        # Assign the topic ID to the paper based on its classification
        paper.topic_id = paper_id_to_topic_id.get(paper.id, None)
        if paper.topic_id is None:
            logging.warning(f"Paper {paper.id} classified under no topic; "
                            "this should not happen with the current logic.")

    # Build works_in from the topics
    build_works_in(scholar_info)

    return scholar_info
