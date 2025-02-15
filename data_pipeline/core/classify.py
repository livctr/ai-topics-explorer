from tqdm import tqdm

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from db_utils import return_conn
from core.data_utils import get_chat_completion


keyword_extraction_template = PromptTemplate(
    input_variables=["max_keywords", "abstract"],
    template=(
        "Extract at most {max_keywords} keywords from the arXiv abstract below, comma-separated. "
        "These keywords should indicate what the paper is about (e.g., topic, problem, method, solution).\n\n"
        "{abstract}"
    )
)

with open("assets/taxonomy.txt", 'r') as f:
    template = f.read() 

abstract_classification_template = PromptTemplate(
    input_variables=["abstract"],
    template=template
)


def extract_keywords(llm: ChatOpenAI, abstract: str, max_keywords: int = 7):
    prompt = keyword_extraction_template.invoke({"max_keywords": max_keywords, "abstract": abstract})
    return get_chat_completion(llm, prompt)


def classify_topic(llm: ChatOpenAI, abstract: str):
    prompt = abstract_classification_template.invoke({"abstract": abstract})
    return get_chat_completion(llm, prompt)


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

    
def classify_paper_into_topic_with_llm(
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_completion_tokens: int = 40,
    limit: int = 1000,
    sample_freq: int = 200,
    cost_per_mil_input_token: float = 0.15,
    cost_per_mil_output_token: float = 0.6
) -> None:
    """
    Classify papers into topics using an LLM and update the database.

    Parameters:
        model (str): LLM model name (default "gpt-4o-mini").
        temperature (float): Sampling temperature (default 0.0).
        max_completion_tokens (int): Max tokens in response (default 40).
        limit (int): Max papers per call
        sample_freq (int): Number of papers per print update (default 200).
        cost_per_mil_input_token (float): Cost per 1e6 input tokens (for printing)
        cost_per_mil_output_token (float): Cost per 1e6 output tokens (for printing)

    Returns:
        None
    """
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens
    )

    fill_in_topic_from_llm_in_db(
        llm,
        limit=limit,
        sample_freq=sample_freq,
        cost_per_mil_input_token=cost_per_mil_input_token,
        cost_per_mil_output_token=cost_per_mil_output_token
    )
