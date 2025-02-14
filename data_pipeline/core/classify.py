from db_utils import return_conn

import warnings

import psycopg2
from tqdm import tqdm

from openai import OpenAI

from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI


keyword_extraction_template = PromptTemplate(
    input_variables=["max_keywords", "abstract"],
    template=(
        "Extract at most {max_keywords} keywords from the arXiv abstract below, comma-separated. "
        "These keywords should indicate what the paper is about (e.g., topic, problem, method, solution).\n\n"
        "{abstract}"
    )
)

abstract_classification_template = PromptTemplate(
    input_variables=["abstract"],
    template=(
"""
CS Topics Taxonomy
```
Theory:Algorithms & Data Structures
Theory:Computation & Complexity Theory
Theory:Cryptography
Theory:Formal Methods
Theory:Game Theory & Economics
Systems: Computer Architecture
Systems: Distributed Systems
Systems: Embedded & Real-Time Systems
Systems: High-Performance & Parallel Computing
Systems: Networking
Systems: Operating Systems
Software:Graphics
Software:Human-Computer Interaction
Software:Programming Languages
Software:Security
Software:Software Engineering
Software:Software+X
Data Management:Big Data
Data Management:Databases
Data Management:Information Retrieval
Data Management:Knowledge Graphs & Information Networks
AI:AI Robustness & Security
AI:Computer Vision
AI:Generative AI
AI:Learning Theory
AI:Multimodal AI
AI:Natural Language Processing
AI:Reinforcement Learning
Interdisciplinary Areas
Interdisciplinary Areas:AI+Art
Interdisciplinary Areas:AI+Business
Interdisciplinary Areas:AI+Education
Interdisciplinary Areas:AI+Healthcare/Medicine
Interdisciplinary Areas:AI+Law
Interdisciplinary Areas:AI+Science
Interdisciplinary Areas:AI+Sustainability
Interdisciplinary Areas:Computational Finance
Interdisciplinary Areas:Quantum Computing
Interdisciplinary Areas:Robotics & Control
```

Classify the provided abstract.

Rules
(1) Domain-Specific AI: If the keywords include terms like "Large Language Model" applied to a specific domain, classify them as `Interdisciplinary Areas:AI+X` (where X represents the domain).
(2) Additional Fine-grained Classification Requirement: For classifications that fall into one of
`Software+X`, `Natural Language Processing`, `Computer Vision`, `Generative AI`, `AI+Science`, 
In addition to classification, assign a fine-grained topic, e.g., "AI:Natural Language Processing:LLM Pre-training", "AI:Computer Vision:Object Detection".

Output Format: colon-delimited classification

{abstract}
"""
    )
)


def get_chat_completion(llm: ChatOpenAI, prompt: str):
    openai_response = llm.invoke(prompt)
    return {
        "content": openai_response.content,
        "input_tokens": openai_response.usage_metadata['input_tokens'],
        "output_tokens": openai_response.usage_metadata['output_tokens']
    }


def extract_keywords(llm: ChatOpenAI, abstract: str, max_keywords: int = 7):
    prompt = keyword_extraction_template.invoke({"max_keywords": max_keywords, "abstract": abstract})
    return get_chat_completion(llm, prompt)


def classify_topic(llm: ChatOpenAI, abstract: str):
    prompt = abstract_classification_template.invoke({"abstract": abstract})
    return get_chat_completion(llm, prompt)


def fill_in_keywords_in_db(limit: int = 10, sample_freq: int = 200):
    """
    Iterates over a limited number of papers without keywords, extracts keywords using extract_keyword,
    and updates the database with the results.

    Parameters:
        limit (int): The maximum number of papers to process.
    """
    conn = return_conn()
    cur = conn.cursor()
    
    # Fetch papers with missing or empty keywords, limiting the number of papers
    cur.execute("""
        SELECT arxiv_id, abstract 
        FROM paper 
        WHERE keywords IS NULL OR keywords = ''
        ORDER BY date DESC
        LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    print(f"Number of papers needing keyword extraction: {len(rows)} ")

    i = 0
    
    # Use tqdm to show progress as we iterate over the papers
    for arxiv_id, abstract in tqdm(rows, desc="Getting keywords"):
        try:
            # Extract keywords using the provided function
            keywords = extract_keyword(abstract)

            if i % sample_freq == 0:
                print(f"Updating {arxiv_id} with keywords: {keywords}")
            i += 1

            # Update the record with the extracted keywords
            cur.execute("""
                UPDATE paper
                SET keywords = %s
                WHERE arxiv_id = %s
            """, (keywords, arxiv_id))
            conn.commit()
        except Exception as e:
            print(f"Error processing arxiv_id {arxiv_id}: {e}")
            # Rollback if there's an error and continue with the next record
            conn.rollback()

    cur.close()
    conn.close()



def fill_in_topic_from_llm_in_db(llm: ChatOpenAI,
                                 limit: int = 10,
                                 sample_freq: int = 200,
                                 cost_per_mil_input_token: float = 0.15,
                                 cost_per_mil_output_token: float = 0.6):
    """
    Iterates over a limited number of papers without a topic_from_llm,
    classifies the abstract using classify_topic, and updates the database with the result.
    
    At every `sample_freq`, prints out the current paper's arxiv_id, classification result,
    and the accumulated cost based on the usage metadata from the LLM.
    
    Parameters:
        llm (ChatOpenAI): An instance of the ChatOpenAI model.
        limit (int): The maximum number of papers to process.
        sample_freq (int): Frequency at which to print progress and cost information.
        cost_per_mil_input_token (float): Cost per 1,000,000 input tokens.
        cost_per_mil_output_token (float): Cost per 1,000,000 output tokens.
    """
    conn = return_conn()
    cur = conn.cursor()
    
    # Fetch papers with missing or empty topic_from_llm, limiting the number of papers.
    cur.execute("""
        SELECT arxiv_id, abstract, title
        FROM paper 
        WHERE topic_from_llm IS NULL OR topic_from_llm = ''
        ORDER BY date DESC
        LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    print(f"Number of papers needing topic classification: {len(rows)}")
    
    accumulated_cost = 0.0
    i = 0

    for arxiv_id, abstract, title in tqdm(rows, desc="Classifying topics"):
        try:
            # Get classification result using our helper function.
            result = classify_topic(llm, abstract)
            topic_from_llm = result["content"]
            input_tokens = result["input_tokens"]
            output_tokens = result["output_tokens"]
            
            # Calculate cost using the token usage from the LLM response.
            cost_input = (input_tokens / 1e6) * cost_per_mil_input_token
            cost_output = (output_tokens / 1e6) * cost_per_mil_output_token
            cost = cost_input + cost_output
            accumulated_cost += cost
            
            if i % sample_freq == 0:
                print(f"Processing arxiv_id {arxiv_id}: `{title}`")
                print(f"  Topic from LLM: {topic_from_llm.strip()}")
                print(f"  Accumulated cost so far: ${accumulated_cost:.4f}")
            i += 1
            
            # Update the record with the classified topic.
            cur.execute("""
                UPDATE paper
                SET topic_from_llm = %s
                WHERE arxiv_id = %s
            """, (topic_from_llm, arxiv_id))
            conn.commit()
        except Exception as e:
            print(f"Error processing arxiv_id {arxiv_id}: {e}")
            conn.rollback()
    
    cur.close()
    conn.close()



# To run the async function:
if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4o-mini",
                     temperature=0.0,
                     max_completion_tokens=40
    )
    cost_per_mil_input_tokens = 0.15  # cached can be x2 cheaper
    cost_per_mil_output_tokens = 0.6

    fill_in_topic_from_llm_in_db(llm,
                                 limit=1000,
                                 sample_freq=50,
                                 cost_per_mil_input_token=cost_per_mil_input_tokens,
                                 cost_per_mil_output_token=cost_per_mil_output_tokens)
