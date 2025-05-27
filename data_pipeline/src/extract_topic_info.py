"""
This module implements an agentic classification workflow for research papers,
extracting their main topics, AI-related status, and subtopics using a state graph approach.
"""

# Standard library imports
import logging
import os
from collections import defaultdict
from functools import partial
from typing import List, Optional, Dict, TypedDict, Sequence, Set, Tuple

# Third-party imports
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Local imports
from src.data_models import Paper, Topic, WorksIn, ScholarInfo, write_scholar_info

# Configuration
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ======================
# Data models in agentic classification
# ======================

class PaperTopicOutput(BaseModel):
    """
    Represents the extracted analysis of a research paper.
    """
    main_topic: str = Field(description="The main academic or research topic/field of the paper (e.g., 'Natural Language Processing', 'Computer Vision', 'Materials Science', 'Neuroscience').")
    is_ai_related: bool = Field(description="True if the paper is directly or indirectly related to Artificial Intelligence, Machine Learning, Deep Learning, or a core AI subfield; otherwise False.")


class SubTopicOutput(BaseModel): # Using LangchainBaseModel as in original
    sub_topic: str = Field(description="The concise subtopic (2-5 words) for the paper within its main topic.")


class PaperClassificationState(TypedDict):
    papers_to_process: Sequence[Paper] # Input batch of papers
    current_paper_index: int # To track progress through the batch
    
    # Data for the current paper being processed
    current_paper_id: Optional[str]
    current_paper_title: Optional[str]
    current_paper_abstract: Optional[str]
    
    # Intermediate results for the current paper
    paper_is_ai_related: Optional[bool]
    paper_main_topic: Optional[str]
    paper_sub_topic: Optional[str]

    # Aggregated results
    classified_papers: List[Dict] # List of classified paper dicts
    
    # Memory for subtopics
    subtopic_memory: Dict[str, List[str]] # {main_topic: [subtopic1, subtopic2, ...]}


# ======================
# Helper functions
# ======================
def load_main_topics(filepath: str = "assets/cold_start.txt") -> List[Dict[str, str]]:
    topics = []
    with open(filepath, 'r') as f:
        content = f.read().strip()
    if not content:
        return topics
    entries = content.split("\n\n")
    for entry in entries:
        lines = entry.strip().split("\n")
        if len(lines) >= 3:
            topic_name = lines[0].replace("Topic: ", "").strip()
            description = lines[1].replace("Description: ", "").strip()
            examples = lines[2].replace("Examples: ", "").strip()
            topics.append({"name": topic_name, "description": description, "examples": examples})
        else:
            logging.warning(f"Malformed entry in topics file: {entry}")

    valid_main_topic_names = [topic['name'] for topic in topics]
    if not valid_main_topic_names:
        # warning print("Warning: valid_main_topic_names is empty. Main topic classification might always default to 'Unclassified'.")
        # Adding a default fallback if the file is missing/empty to prevent downstream errors with empty list
        valid_main_topic_names.append("Unclassified")
    return valid_main_topic_names


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def get_closest_topic_index(main_topic, valid_main_topic_names):
    main_topic_lower = main_topic.lower()
    valid_names_lower = [name.lower() for name in valid_main_topic_names]

    min_distance = float('inf')
    closest_index = -1

    for i, name in enumerate(valid_names_lower):
        dist = levenshtein_distance(main_topic_lower, name)
        if dist < min_distance:
            min_distance = dist
            closest_index = i

    return closest_index


def build_topics_from_state(scholar_info: ScholarInfo, clean_final_state: PaperClassificationState) -> None:
    """
    Populates scholar_info.topics with Topic objects derived from clean_final_state,
    and updates scholar_info.papers with their respective topic_id.
    """
    def _is_valid_topic_name(name: Optional[str]) -> bool:
        """Checks if a topic name is valid for processing (not None, empty, or common placeholders/errors)."""
        if name is None:
            return False
        
        name_stripped = name.strip()
        if not name_stripped: # Empty string after stripping
            return False
            
        name_lower = name_stripped.lower()
        # Filter out common placeholders or error indicators
        if name_lower in ["n/a", "none", "unclassified", "error in generation", "error"]: # "Unclassified" itself is a valid topic name, but check if other error strings are present.
            # Let's refine: "Unclassified" is a valid topic. Error strings are not.
            if name_lower == "unclassified":
                return True # "Unclassified" is a valid topic name handled specifically
            if "error in generation" in name_lower or name_lower == "error":
                return False
            if name_lower in ["n/a", "none"]: # Common placeholders for no topic
                return False
        return True

    next_topic_id = 0  # Start topic IDs from 0

    # Clear existing topics in scholar_info to prevent duplication if function is re-run.
    # This assumes this function is responsible for constructing the entire topics list.
    scholar_info.topics = [] 
    
    # Intermediate mappings to store processed topic names and their assigned IDs
    processed_main_topics: Dict[str, int] = {}  # Maps main_topic_name to its ID
    processed_sub_topics: Dict[Tuple[str, str], int] = {}  # Maps (main_topic_name, sub_topic_name) to its ID

    # --- Step 1: Gather all unique valid topic names and subtopic pairs ---
    all_main_topic_names: Set[str] = set()
    all_sub_topic_pairs: Set[Tuple[str, str]] = set() # Stores (main_topic_name, sub_topic_name)

    # Extract topics from subtopic_memory
    for main_name_mem, sub_names_list in clean_final_state.get("subtopic_memory", {}).items():
        # Main topic from memory key must be valid
        if _is_valid_topic_name(main_name_mem) or main_name_mem == "Unclassified": # Allow "Unclassified" as a main topic key
            all_main_topic_names.add(main_name_mem)
            for sub_name_mem in sub_names_list:
                if _is_valid_topic_name(sub_name_mem): # Subtopic from memory list must be valid
                    all_sub_topic_pairs.add((main_name_mem, sub_name_mem))

    # Extract topics from classified_papers
    for classified_paper_dict in clean_final_state.get("classified_papers", []):
        is_ai_related = classified_paper_dict.get("is_ai_related", False)
        main_name_cp = classified_paper_dict.get("main_topic")
        sub_name_cp = classified_paper_dict.get("sub_topic")

        # Allow "Unclassified" as a valid main topic from classified papers
        if is_ai_related and (_is_valid_topic_name(main_name_cp) or main_name_cp == "Unclassified"):
            all_main_topic_names.add(main_name_cp)
            # If there's a valid sub_topic, it forms a pair with its main_topic
            if _is_valid_topic_name(sub_name_cp) and (_is_valid_topic_name(main_name_cp) or main_name_cp == "Unclassified"):
                 all_sub_topic_pairs.add((main_name_cp, sub_name_cp))
    
    # --- Step 2: Create Main Topic Objects ---
    for main_topic_name in sorted(list(all_main_topic_names)): # Sort for deterministic ID assignment
        if main_topic_name not in processed_main_topics:
            # User rule: "main topic should ... be is_leaf false".
            # "Unclassified" is logically a leaf.
            is_leaf_status = True if main_topic_name == "Unclassified" else False
            
            try:
                main_topic_obj = Topic(
                    id=next_topic_id,
                    name=main_topic_name,
                    parent_id=None, # Main topics have no parent
                    level=1,
                    is_leaf=is_leaf_status
                )
                scholar_info.topics.append(main_topic_obj)
                processed_main_topics[main_topic_name] = next_topic_id
                next_topic_id += 1
            except ValueError as e: 
                logging.error(f"Error creating main Topic object for '{main_topic_name}': {e}")

    # --- Step 3: Create Subtopic Objects ---
    for main_topic_name, sub_topic_name in sorted(list(all_sub_topic_pairs)): # Sort for deterministic ID assignment
        if (main_topic_name, sub_topic_name) not in processed_sub_topics:
            # Parent main topic must exist and have been processed
            if main_topic_name in processed_main_topics:
                parent_topic_id = processed_main_topics[main_topic_name]
                try:
                    sub_topic_obj = Topic(
                        id=next_topic_id,
                        name=sub_topic_name,
                        parent_id=parent_topic_id,
                        level=2,
                        is_leaf=True # Subtopics are always leaves as per problem description
                    )
                    scholar_info.topics.append(sub_topic_obj)
                    processed_sub_topics[(main_topic_name, sub_topic_name)] = next_topic_id
                    next_topic_id += 1
                except ValueError as e: 
                    logging.error(f"Error creating sub-Topic object for '({main_topic_name}, {sub_topic_name})': {e}")
            else:
                logging.warning(f"Parent main topic '{main_topic_name}' not found in processed_main_topics for subtopic '{sub_topic_name}'. Skipping subtopic.")

    # --- Step 4: Update scholar_info.papers (both topic_id and if in list) ---
    papers_in_scholar_info_map: Dict[str, Paper] = {p.id: p for p in scholar_info.papers}
    scholar_info.papers = []  # Clear existing papers, only include AI related ones

    for classified_paper_dict in clean_final_state.get("classified_papers", []):
        paper_id_str = classified_paper_dict.get("id")
        main_name_cp = classified_paper_dict.get("main_topic")
        is_ai_related = classified_paper_dict.get("is_ai_related", False)
        sub_name_cp = classified_paper_dict.get("sub_topic")

        paper_to_update = papers_in_scholar_info_map.get(paper_id_str)
        if not paper_to_update:
            logging.warning(f"Paper ID '{paper_id_str}' from classified_papers not found. Cannot update its topic_id.")
            continue
    
        if not is_ai_related:
            logging.info(f"Paper ID '{paper_id_str}' is not AI-related. Skipping insert.")
        
        assigned_topic_id: Optional[int] = None

        # Check if main_name_cp itself is valid or "Unclassified" before looking up pairs
        main_topic_is_usable = _is_valid_topic_name(main_name_cp) or main_name_cp == "Unclassified"

        # Try to assign the most specific topic ID (subtopic) first
        if main_topic_is_usable and _is_valid_topic_name(sub_name_cp):
            assigned_topic_id = processed_sub_topics.get((main_name_cp, sub_name_cp))
        
        # If no subtopic ID was assigned (or subtopic was not valid), try main topic ID
        if assigned_topic_id is None and main_topic_is_usable:
            assigned_topic_id = processed_main_topics.get(main_name_cp)
        
        if assigned_topic_id is not None:
            paper_to_update.topic_id = assigned_topic_id
            scholar_info.papers.append(paper_to_update)
        else:
            if main_topic_is_usable or _is_valid_topic_name(sub_name_cp): # Log if there were topic names but they didn't map to an ID.
                 logging.warning(f"No processed topic ID found for paper '{paper_id_str}' (Main: '{main_name_cp}', Sub: '{sub_name_cp}'). Its topic_id will remain None.")


def build_works_in(
    scholar_info: ScholarInfo,
) -> None:
    researchers = {r.id for r in scholar_info.researchers}
    topics = {t.id: t for t in scholar_info.topics}
    works_in = defaultdict(float)

    scholar_info.works_in = []  # Clear past works_in data

    for paper in scholar_info.papers:
        if paper.topic_id is None:
            logging.warning(f"Paper ID '{paper.id}' has no topic_id. Skipping works_in assignment.")
            continue

        for r in paper.researcher_ids:
            if r not in researchers:
                logging.warning(f"Researcher ID '{r}' for paper ID '{paper.id}' not found in researchers. Skipping works_in assignment.")
            else:
                score = 1.0 / len(paper.researcher_ids)

                topic_id = paper.topic_id
                while topic_id is not None:
                    works_in[(r, topic_id)] += score
                    topic_id = topics[topic_id].parent_id if topic_id in topics else None

    for (r, topic_id), score in works_in.items():
        if r in researchers and topic_id in topics:
            scholar_info.works_in.append(
                WorksIn(
                    researcher_id=r,
                    topic_id=topic_id,
                    score=score
                )
            )
                


# ======================
# Graph node definitions
# ======================
def start_processing_next_paper(state: PaperClassificationState) -> Dict:
    """Sets up the data for the next paper to be processed from the batch."""
    logging.info("--- STARTING NEXT PAPER ---")
    idx = state.get("current_paper_index", 0)
    papers = state["papers_to_process"]
    
    if idx >= len(papers):
        return {"current_paper_id": None} # Signal to end

    paper_data = papers[idx]
    logging.info(f"Processing paper {idx + 1}/{len(papers)}: ID {paper_data.id}, Title: {paper_data.title}")
    
    # The original AttributeError implies paper_data (a Paper object) doesn't have 'abstract'.
    # Given the Pydantic model, this should not happen if Paper instances are correctly created.
    # No change here as paper_data.abstract should be valid.
    current_abstract = paper_data.abstract if paper_data.abstract else None

    return {
        "current_paper_id": paper_data.id,
        "current_paper_title": paper_data.title,
        "current_paper_abstract": current_abstract,
        "paper_is_ai_related": None,  # Clear previous paper data
        "paper_main_topic": None,
        "paper_sub_topic": None,
    }


def extract_paper_fields(state: PaperClassificationState, llm: ChatOpenAI, valid_main_topic_names: List[str]) -> Dict:
    """Extracts the main topic and AI-related status from the paper's title and abstract."""
    title = state["current_paper_title"]
    abstract = state["current_paper_abstract"]

    if not title: # Basic check
        logging.warning("Warning: Title is missing for paper. Skipping extraction.")
        return {
            "paper_is_ai_related": False,
            "paper_main_topic": "Unclassified",
        }

    parser = PydanticOutputParser(pydantic_object=PaperTopicOutput)

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an expert research assistant. Your task is to analyze a research paper "
            "and extract specific information in a structured JSON format.\n"
            "Valid main topics are: {valid_topics}\n"
            "{format_instructions}"
        ),
        HumanMessagePromptTemplate.from_template(
            "Analyze the following paper based on its title and abstract. "
            "Identify its main academic topic from the provided list, "
            "and state whether it is related to Artificial Intelligence (AI). "
            "If the main focus of the paper is to apply AI to some application, classify it as 'AI Application'. \n\n"
            "Title: {title}\n"
            "Abstract: {abstract}\n\n"
        )
    ])

    valid_topics_str = ", ".join(valid_main_topic_names) if valid_main_topic_names else "General Science"

    chain = prompt_template.partial(
        format_instructions=parser.get_format_instructions(),
        valid_topics=valid_topics_str
    ) | llm | parser

    try:
        # Use fuzzy matching to find the closest main topic
        response = chain.invoke({"title": title, "abstract": abstract if abstract else 'N/A'})
        main_topic = response.main_topic.strip()
        index = get_closest_topic_index(main_topic, valid_main_topic_names)
        if index != -1:
            main_topic = valid_main_topic_names[index]
        else:
            logging.warning(f"LLM returned main topic '{main_topic}' which is not in the predefined list {valid_main_topic_names}. Defaulting to 'Unclassified'.")
            main_topic = "Unclassified"
    except Exception as e:
        logging.warning(f"Error in paper description extraction for title '{title}': {e}. Defaulting.")
        response = PaperTopicOutput(
            main_topic="Unclassified",
            is_ai_related=False
        )
        main_topic = "Unclassified"

    logging.info(f"  Main Topic: {main_topic}")
    logging.info(f"  Is AI Related: {response.is_ai_related}")
    return {
        "paper_is_ai_related": response.is_ai_related,
        "paper_main_topic": main_topic, # Use the validated/defaulted main_topic
    }


def classify_or_generate_subtopic(state: PaperClassificationState, llm: ChatOpenAI) -> Dict:
    """Classifies into an existing subtopic or generates a new one,
    using paper's title, abstract, main topic, and subtopic memory."""
    
    # Retrieve necessary data from the state
    title = state["current_paper_title"]
    abstract = state["current_paper_abstract"] # This is the original abstract
    is_ai_related = state["paper_is_ai_related"]
    main_topic = state["paper_main_topic"]
    subtopic_memory = state.get("subtopic_memory", {})

    # Keep the initial checks
    if not main_topic or main_topic == "Unclassified":
        logging.info(f"  Main topic is '{main_topic}'. Skipping subtopic classification.")
        return {"paper_sub_topic": "N/A"} # No change to subtopic_memory in this path

    if is_ai_related is None or not is_ai_related:
        logging.info("  Paper is not AI-related. Skipping subtopic classification.")
        return {"paper_sub_topic": "N/A"} # No change to subtopic_memory in this path

    # Prepare the part of the prompt that lists existing subtopics
    existing_subtopics_for_main_topic = subtopic_memory.get(main_topic, [])
    
    if existing_subtopics_for_main_topic:
        subtopics_list_str = "\n- ".join(existing_subtopics_for_main_topic)
        subtopics_prompt_part = (
            f"Consider these existing subtopics for '{main_topic}':\n"
            f"- {subtopics_list_str}\n"
            "If the paper (detailed below) fits well into one of these, please select it. "
            "Otherwise, generate a new, concise subtopic for the paper (2-5 words) based on its content. "
            "Aim to reuse existing subtopics where appropriate to maintain a limited set (ideally around 10 subtopics per main topic in total eventually). "
            "Try to keep the subtopic as general as possible."
        )
    else:
        subtopics_prompt_part = (
            "There are no existing subtopics for this main topic yet. "
            "Please generate a new, concise subtopic (2-5 words) for the paper (detailed below) based on its content."
        )

    # Define the prompt template, now including title and abstract
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an expert research assistant. Your task is to define a specific subtopic for a research paper "
            "within its assigned main topic, based on the paper's title and abstract. Subtopics should be concise (2-5 words)."
        ),
        HumanMessagePromptTemplate.from_template(
            "Analyze the following paper details:\n"
            "Title: {title}\n"
            "Abstract: {abstract}\n\n" # Abstract will be 'N/A' if originally None
            "Main Topic: '{main_topic}'\n\n"
            "{subtopics_prompt_part}\n\n"
            "What is the most fitting subtopic for this paper? If generating a new one, ensure it's distinct and specific. "
            "Return only the subtopic name."
        )
    ])

    structured_llm_sub_topic = llm.with_structured_output(SubTopicOutput)
    chain = prompt_template | structured_llm_sub_topic
    
    sub_topic_str = "N/A" # Default value
    
    # Ensure abstract is a string for the prompt
    abstract_for_prompt = abstract if abstract else "N/A"

    try:
        response = chain.invoke({
            "title": title,
            "abstract": abstract_for_prompt,
            "main_topic": main_topic,
            "subtopics_prompt_part": subtopics_prompt_part
        })
        sub_topic_str = response.sub_topic.strip()
        logging.info(f"  Generated/Selected Subtopic for '{main_topic}': {sub_topic_str}")

        # Update subtopic memory (this logic remains the same)
        if main_topic not in subtopic_memory:
            subtopic_memory[main_topic] = []
        
        # Add new subtopic to memory if it's not already there and not a placeholder/error
        if sub_topic_str not in subtopic_memory[main_topic] and sub_topic_str != "N/A" and "Error in generation" not in sub_topic_str:
            if len(subtopic_memory[main_topic]) < 15: # Soft limit
                subtopic_memory[main_topic].append(sub_topic_str)
            else:
                logging.info(f"  Note: Subtopic count for '{main_topic}' is high. LLM suggested '{sub_topic_str}' but not adding to memory to control growth.")
    
    except Exception as e:
        logging.error(f"  Error in subtopic generation for main topic '{main_topic}' (Title: '{title}'): {e}. Defaulting subtopic.")
        sub_topic_str = "Error in generation"

    return {"paper_sub_topic": sub_topic_str, "subtopic_memory": subtopic_memory}


def aggregate_results(state: PaperClassificationState) -> Dict:
    """Aggregates the classification for the current paper and increments index."""
    logging.info("--- AGGREGATING RESULTS ---")
    classified_papers = state.get("classified_papers", [])

    current_classification = {
        "id": state["current_paper_id"],
        "title": state["current_paper_title"],
        "abstract": state["current_paper_abstract"],
        "is_ai_related": state["paper_is_ai_related"],
        "main_topic": state["paper_main_topic"], # Corrected key
        "sub_topic": state["paper_sub_topic"]    # Corrected key
    }
    classified_papers.append(current_classification)
    
    next_index = state.get("current_paper_index", 0) + 1

    logging.info(f"  Finished paper ID {state['current_paper_id']}. Main: {state['paper_main_topic']}, Sub: {state['paper_sub_topic']}")
    return {"classified_papers": classified_papers, "current_paper_index": next_index}


# ======================
# State graph definition
# ======================
def build_workflow() -> StateGraph:

    def should_continue_processing(state: PaperClassificationState) -> str:
        """Determines if there are more papers to process."""
        if state.get("current_paper_id") is None: 
            logging.info("--- BATCH PROCESSING COMPLETE ---")
            return "finish_processing"
        else:
            # The key for the next node should match the actual node name in add_conditional_edges
            return "extract_paper_fields_node"

    # Initialize LLM and main topics for the workflow nodes
    llm = ChatOpenAI(temperature=0, model_name="gpt-4.1-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))
    main_topics = load_main_topics()
    extract_paper_fields_node = partial(extract_paper_fields, llm=llm, valid_main_topic_names=main_topics)
    classify_subtopic_node = partial(classify_or_generate_subtopic, llm=llm)

    workflow = StateGraph(PaperClassificationState)

    # Add nodes
    workflow.add_node("start_next_paper_node", start_processing_next_paper)
    workflow.add_node("extract_paper_fields_node", extract_paper_fields_node)
    workflow.add_node("classify_subtopic_node", classify_subtopic_node)
    workflow.add_node("aggregate_results_node", aggregate_results)

    # Set entry point
    workflow.set_entry_point("start_next_paper_node")

    # Add edges
    workflow.add_conditional_edges(
        "start_next_paper_node",
        should_continue_processing,
        {
            "extract_paper_fields_node": "extract_paper_fields_node", # Path if should_continue returns this string
            "finish_processing": END 
        }
    )
    workflow.add_edge("extract_paper_fields_node", "classify_subtopic_node")
    workflow.add_edge("classify_subtopic_node", "aggregate_results_node")
    workflow.add_edge("aggregate_results_node", "start_next_paper_node")

    return workflow


def run_agentic_classification(scholar_info: ScholarInfo, rebuild_topics_if_exists: bool = True) -> ScholarInfo:
    """
    Runs the agentic classification workflow on a list of papers.
    """
    # Initialize the state
    papers = scholar_info.papers
    topics = scholar_info.topics
    if not rebuild_topics_if_exists and topics:
        logging.info("Topics already exist. No need to build them again.")
        return
    initial_state: PaperClassificationState = {
        "papers_to_process": papers,
        "current_paper_index": 0,
        "current_paper_id": None,
        "current_paper_title": None,
        "current_paper_abstract": None,
        "paper_is_ai_related": None,
        "paper_main_topic": None,
        "paper_sub_topic": None,
        "classified_papers": [],
        "subtopic_memory": {} 
    }

    # Build the workflow
    workflow = build_workflow()
    app = workflow.compile()

    # Run the workflow
    config = {"recursion_limit": len(papers) * 5 + 50} 
    clean_final_state = app.invoke(initial_state, config=config)

    # Build topics from the final state
    build_topics_from_state(scholar_info, clean_final_state)

    # Build works_in from the topics
    build_works_in(scholar_info)

    # Write the updated scholar_info to file
    write_scholar_info(scholar_info)

    return scholar_info

