import os
from types import SimpleNamespace
from pathlib import Path

import pytest

from src.data_models import Paper
from src.extract_topic_info import TopicsTree, classify_paper

import pytest
from src.extract_topic_info import TopicsTree

def test_topics_tree_empty():
    tree = TopicsTree()
    assert tree.query_subtopics(()) == set()

def test_add_and_query():
    tree = TopicsTree()
    tree.add_paper(("Computer Vision",), "paper1")
    assert tree.query_subtopics(()) == {"Computer Vision"}

    tree.add_paper(("Computer Vision", "Object Segmentation"), "abcdef")
    assert tree.query_subtopics(("Computer Vision",)) == {"Object Segmentation"}

def test_copy_tree():
    tree = TopicsTree()
    tree.add_paper(("A", "B"), "p1")
    tree_copy = TopicsTree(tree)
    assert tree_copy.query_subtopics(("A",)) == {"B"}


class DummyLLM:
    """
    A fake LLM that returns scripted responses for each .invoke() call.
    """
    def __init__(self, scripted):
        self._queue = list(scripted)

    def invoke(self, *args, **kwargs):
        if not self._queue:
            raise RuntimeError("No more scripted LLM responses!")
        return SimpleNamespace(content=self._queue.pop(0))


def test_pairings_and_topic_info():
    tree = TopicsTree()
    # Build simple hierarchy: A -> B
    tree.add_paper(("A",), "p1")
    tree.add_paper(("A", "B"), "p2")

    paper_map, hierarchy = tree.get_pairings_and_topic_info()

    # Expect mapping of paper IDs to a single topic ID (leaf)
    assert paper_map == {"p1": 1, "p2": 2}

    # Expect hierarchy entries only for nodes with papers
    assert hierarchy == [
        (1, "A", None, 1, False),  # A has child B
        (2, "B", 1, 2, True),      # B is leaf
    ]


def test_classify_paper_merge_and_routing():
    # Scripted LLM outputs in the exact order classify_paper will call:
    script = [
        "Other",                                     # p0 -> Other
        "AI Applications", "AI for Healthcare",    # p1
        "Computer Vision", "Object Detection & Recognition",  # p2
        "Computer Vision", "Object Detection & Recognition",  # p3
        "Computer Vision", "None: Image Segmentation", "Object Detection & Recognition",  # p4
        "Computer Vision", "Depth Estimation",     # p5
    ]
    dummy = DummyLLM(script)
    llm_fn = dummy.invoke

    # Load cold-start topics
    tree = TopicsTree()
    tree.load_cold_start(Path("./assets/cold_start.txt"))

    # Define papers p0–p5
    papers = [
        Paper(id="p0", title="Refrigerator Shelf Arrangement Optimisation", abstract="", date="2024-01-01"),
        Paper(id="p1", title="Deep Learning for Radiology Image Segmentation", abstract="", date="2024-02-02"),
        Paper(id="p2", title="YOLO-v9000: Real-Time Object Detection Re-Redux", abstract="", date="2024-03-03"),
        Paper(id="p3", title="A Faster SIFT for Mobile Vision", abstract="", date="2024-04-04"),
        Paper(id="p4", title="Semantic Masking with Diffusion Models", abstract="", date="2024-05-05"),
        Paper(id="p5", title="Monocular Depth Estimation by Large ViTs", abstract="", date="2024-07-07"),
    ]

    # Classify each paper with max_subtopics_per_main_topic=1 to trigger merge logic
    for paper in papers:
        classify_paper(
            paper=paper,
            topics_tree=tree,
            llm=llm_fn,
            max_subtopics_per_main_topic=1,
        )
    
    # Verify that papers were routed correctly by inspecting node.papers
    other_node = tree.root.find_node(("Other",))
    assert other_node and "p0" in other_node.papers

    cv_node  = tree.root.find_node(("Computer Vision",))
    seg_node = tree.root.find_node(("Computer Vision", "Image Segmentation"))
    de_node  = tree.root.find_node(("Computer Vision", "Depth Estimation"))
    assert cv_node and seg_node and de_node

    # p2, p3, p4 should be under Image Segmentation after merge
    for pid in ["p2", "p3", "p4"]:
        assert pid in seg_node.papers

    # p5 should be under Depth Estimation
    assert "p5" in de_node.papers


if __name__ == "__main__":
    test_add_and_query()
    test_topics_tree_empty()
    test_copy_tree()
    test_pairings_and_topic_info()
    test_classify_paper_merge_and_routing()
    print("All tests passed.")
