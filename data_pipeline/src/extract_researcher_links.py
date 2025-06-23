import os
import logging
from typing import List, Optional, Dict, Tuple
from pydantic import BaseModel, Field
from datetime import date

# Langchain imports
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Tavily client
from tavily import TavilyClient

from src.data_models import (
    ResearcherLink, Researcher, ScholarInfo
)


class ExtractedInfo(BaseModel):
    """Pydantic model for parsing LLM output for researcher details."""
    homepage_index: Optional[int] = Field(None, description="1-based index from search results text. Use null if not applicable/ambiguous.")
    primary_affiliation: Optional[str] = Field(None, description="Single primary affiliation. Use null if not applicable/ambiguous.")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')


def search_author(name: str, tavily_client: TavilyClient, max_results: int = 5) -> Tuple[List[Dict], str]:
    """
    Performs a Tavily search for an AI researcher by name.

    Returns:
        A tuple containing the raw search results (list of dictionaries) 
        and a formatted text string of these results.
    """
    query = f"{name} AI researcher"
    logger.info(f"Searching for author: {name} with query: '{query}'")
    
    try:
        search_response = tavily_client.search(
            query=query,
            search_depth="basic", 
            num_results=max_results,
        )
        raw_results = search_response.get("results", [])
    except Exception as e:
        logger.error(f"Tavily search failed for '{name}': {e}")
        return [], "Search failed or produced an error."

    if not raw_results:
        logger.warning(f"No search results found for author: {name}")
        return [], "No results found."
    
    results_text_parts = []
    for i, result in enumerate(raw_results):
        title = result.get('title', 'N/A')
        url = result.get('url', 'N/A')
        content = result.get('content', 'N/A')
        results_text_parts.append(f"{i+1}. {title} - {url}\n{content}")
    results_text = "\n\n".join(results_text_parts) # Use double newline for better separation
    
    return raw_results, results_text


def run_researcher_info_extraction(
    rll: List[ResearcherLink],
    scholar_info: ScholarInfo,
    max_update: int = 100
) -> List[ResearcherLink]:
    """
    Extracts homepage URL and primary affiliation for researchers using an LLM,
    based on a waterfall method across Level 1 topics.
    """
    current_search_date = date.today().isoformat()
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.critical("OPENAI_API_KEY not found in environment variables. LLM calls will fail.")
        # Depending on strictness, could raise error or return immediately
        return [link.model_copy(deep=True) for link in rll]


    llm = ChatOpenAI(
        model="gpt-4.1-mini", # Using a common and recent model
        temperature=0.0,
        max_tokens=100, # Max tokens for the completion (response from LLM)
        openai_api_key=openai_api_key
    )

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        logger.critical("TAVILY_API_KEY not found in environment variables. Search functionality will fail.")
        return [link.model_copy(deep=True) for link in rll]
    tavily_client = TavilyClient(api_key=tavily_api_key)

    output_rll = [link.model_copy(deep=True) for link in rll]
    processed_researcher_ids = {link.id for link in output_rll}
    
    researcher_map: Dict[int, Researcher] = {
        r.id: r for r in scholar_info.researchers if r.id is not None and r.h_index is not None and r.h_index >= 1
    }

    level1_topics = [t for t in scholar_info.topics if t.level == 1]
    if not level1_topics:
        logger.info("No level 1 topics found. Returning initial researcher links.")
        return output_rll

    topic_candidate_queues: Dict[int, List[Tuple[float, int]]] = {}
    for topic in level1_topics:
        candidates_for_topic: List[Tuple[float, int]] = []
        for work_entry in scholar_info.works_in:
            if work_entry.topic_id == topic.id:
                if work_entry.researcher_id in researcher_map:
                    candidates_for_topic.append((work_entry.score, work_entry.researcher_id))
        
        candidates_for_topic.sort(key=lambda x: (-x[0], x[1])) # Sort by score desc, then ID asc
        if candidates_for_topic:
            topic_candidate_queues[topic.id] = candidates_for_topic
    
    if not topic_candidate_queues:
        logger.info("No candidate researchers w/ non-null h-index >= 1 found in any level 1 topic queues after filtering.")
        return output_rll

    updates_done = 0
    queue_indices = {topic_id: 0 for topic_id in topic_candidate_queues}
    active_topic_ids_ordered = [t.id for t in level1_topics if t.id in topic_candidate_queues]

    # Setup PydanticOutputParser and PromptTemplate once
    parser = PydanticOutputParser(pydantic_object=ExtractedInfo)
    
    system_prompt_text = (
        "You are an AI assistant. From the provided search results:\n"
        "1. Identify the 1-based index number of the entry that represents the researcher's primary homepage. "
        "If no single, clear homepage can be identified, or if the results are ambiguous "
        "(e.g., multiple distinct researchers with the same name), provide null for 'homepage_index'.\n"
        "2. Extract their single primary affiliation. If no affiliation is found or it's ambiguous, "
        "provide null for 'primary_affiliation'.\n"
        "Respond strictly in the JSON format described below:\n{format_instructions}"
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt_text),
        HumanMessagePromptTemplate.from_template("Search results for {researcher_name}:\n{search_results_text}")
    ])
    
    # LCEL Chain
    chain = prompt_template | llm | parser

    while updates_done < max_update:
        processed_in_this_pass = False
        for topic_id in active_topic_ids_ordered:
            if updates_done >= max_update:
                break

            current_idx = queue_indices[topic_id]
            candidates_list = topic_candidate_queues[topic_id]

            if current_idx < len(candidates_list):
                score, researcher_id = candidates_list[current_idx]
                queue_indices[topic_id] += 1 # Advance index for this topic for the next round

                if researcher_id not in processed_researcher_ids:
                    researcher = researcher_map.get(researcher_id)
                    if not researcher:
                        logger.warning(f"Researcher ID {researcher_id} not found in researcher_map at processing time. Skipping.")
                        continue

                    logger.info(f"Processing researcher: {researcher.name} (ID: {researcher_id}), Score: {score}, From Topic ID: {topic_id}")

                    raw_search_results, formatted_search_text = search_author(researcher.name, tavily_client)

                    if not raw_search_results:
                        logger.warning(f"No search results obtained for {researcher.name} via Tavily. Skipping LLM extraction.")
                    else:
                        try:
                            response_data: ExtractedInfo = chain.invoke({
                                "researcher_name": researcher.name, 
                                "search_results_text": formatted_search_text,
                                "format_instructions": parser.get_format_instructions()
                            })
                            
                            actual_homepage_url: Optional[str] = None
                            if response_data.homepage_index is not None and response_data.homepage_index > 0:
                                h_idx = response_data.homepage_index - 1 # Adjust for 0-based list indexing
                                if 0 <= h_idx < len(raw_search_results):
                                    actual_homepage_url = raw_search_results[h_idx].get('url')
                                    logger.info(f"Extracted homepage index {response_data.homepage_index} for {researcher.name}, URL: {actual_homepage_url}")
                                else:
                                    logger.warning(f"LLM returned out-of-bounds homepage_index {response_data.homepage_index} for {researcher.name}. Max index: {len(raw_search_results)}")
                            elif response_data.homepage_index is not None: # Handles 0 or other non-positive values if LLM returns them
                                logger.info(f"LLM indicated no specific homepage (index: {response_data.homepage_index}) for {researcher.name}.")
                            
                            affiliations_list = []
                            # Add affiliation if present and not the literal string "N/A" (case-insensitive)
                            if response_data.primary_affiliation and response_data.primary_affiliation.strip().upper() != "N/A":
                                affiliations_list.append(response_data.primary_affiliation.strip())
                            
                            new_link = ResearcherLink(
                                id=researcher_id, 
                                name=researcher.name,
                                homepage=actual_homepage_url,
                                affiliations=affiliations_list,
                                search_date=current_search_date
                            )
                            output_rll.append(new_link)
                            logger.info(f"Processed info for {researcher.name}: Homepage Index='{response_data.homepage_index}', Primary Affiliation='{response_data.primary_affiliation}'")

                        except Exception as e:
                            logger.error(f"Error during LLM chain processing or data handling for {researcher.name}: {e}", exc_info=True)
                    
                    processed_researcher_ids.add(researcher_id) # Mark as processed regardless of LLM success
                    updates_done += 1
                    processed_in_this_pass = True
            
        if not processed_in_this_pass:
            logger.info("A full pass completed with no new researchers processed (all queues exhausted or remaining candidates already processed). Stopping.")
            break
        
        all_queues_truly_exhausted = all(
            queue_indices[topic_id_check] >= len(topic_candidate_queues[topic_id_check])
            for topic_id_check in active_topic_ids_ordered
        )
        if all_queues_truly_exhausted:
            logger.info("All topic candidate queues are fully exhausted.")
            break
            
    logger.info(f"Researcher info extraction finished. Total new researchers processed in this run: {updates_done}")
    return output_rll
