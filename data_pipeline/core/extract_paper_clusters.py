"""
Two-tiered k-means clustering based on embeddings.
"""

import json

from tqdm import tqdm

from core.data_utils import EntryExtractor, FILTERED_PATH, TOPICS_PATH

import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel

import torch
import numpy as np
from sklearn.cluster import KMeans

import random
from typing import List, Optional

from langchain_openai import ChatOpenAI
from core.data_utils import get_chat_completion
from db_utils import return_conn
import argparse

from dotenv import load_dotenv
load_dotenv()

accumulated_cost = 0.0



def build_embeddings(batch_size: int = 32):
    """
    Build embeddings for documents and return the embeddings, arxiv_ids, and mappings of ids to titles and abstracts.
    
    Args:
        batch_size (int): Batch size for processing documents.
        
    Returns:
        tuple: (embeddings, arxiv_ids, id2title, id2abstract)
    """
    def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-1.5B-instruct', trust_remote_code=True)
    model = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen2-1.5B-instruct', trust_remote_code=True)
    # Move model to GPU
    model = model.to(device)

    embeddings_list = []
    arxiv_ids = []
    id2title = {}
    id2abstract = {}
    
    def process_batch(docs, ids):
        """Process a batch of documents through the model and add results to global lists."""
        if not docs:
            return
            
        batch_dict = tokenizer(docs, max_length=8192, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
        
        with torch.no_grad():
            outputs = model(**batch_dict)
            batch_embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            
        embeddings_list.append(batch_embeddings.cpu())
        arxiv_ids.extend(ids)
    
    num_lines = sum(1 for _ in open(FILTERED_PATH))

    # Process documents in batches
    with open(FILTERED_PATH, 'r') as f:
        documents = []
        batch_ids = []
        batch_count = 0

        for line in tqdm(f, desc="Processing documents", total=num_lines):
            try:
                entry_data = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            arxiv_id = EntryExtractor.extract_id(entry_data)
            title = EntryExtractor.extract_title(entry_data)
            abstract = EntryExtractor.extract_abstract(entry_data)
            
            # Store title and abstract in dictionaries
            id2title[arxiv_id] = title
            id2abstract[arxiv_id] = abstract
            
            txt = title + "\n\n" + abstract
            documents.append(txt)
            batch_ids.append(arxiv_id)
            
            # When we reach batch size, process the batch
            if len(documents) >= batch_size:
                process_batch(documents, batch_ids)
                
                # Reset for next batch
                documents = []
                batch_ids = []
                batch_count += 1

        # Process any remaining documents
        process_batch(documents, batch_ids)

    # Concatenate all embeddings
    embeddings = torch.cat(embeddings_list, dim=0)
    return embeddings, arxiv_ids, id2title, id2abstract



def cluster_arxiv_embeddings(embeddings: np.ndarray, arxiv_ids: np.ndarray, 
                             n_clusters: int = 100, n_superclusters: int = 15, 
                             random_state: int = 42):
    """
    Clusters embeddings into n_clusters clusters, then clusters the cluster centers into n_superclusters.

    Args:
        embeddings (np.ndarray): An N x d matrix.
        arxiv_ids (np.ndarray): An array of length N where each entry corresponds to the arxiv_id for the embedding.
        n_clusters (int): Number of clusters for the first k-means.
        n_superclusters (int): Number of super clusters for the second k-means.
        random_state (int): Seed for reproducibility.
    
    Returns:
        supercluster_to_clusters (dict): Maps super-cluster ID to a list of cluster IDs.
        cluster_to_arxiv (dict): Maps cluster ID (from first clustering) to a list of arxiv_ids in that cluster.
    """
    # First spherical k-means: cluster the embeddings into n_clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Normalize the cluster centers to be on the unit sphere
    cluster_centers = kmeans.cluster_centers_
    cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)
    
    # Map cluster id -> arxiv_ids
    cluster_to_arxiv = {i: [] for i in range(n_clusters)}
    for idx, cluster_id in enumerate(cluster_labels):
        cluster_to_arxiv[cluster_id].append(arxiv_ids[idx])
    
    # Second spherical k-means: cluster the normalized cluster centers into n_superclusters
    super_kmeans = KMeans(n_clusters=n_superclusters, random_state=random_state)
    super_labels = super_kmeans.fit_predict(cluster_centers)

    # Normalize the supercluster centers (not needed for the return value but for consistency)
    super_centers = super_kmeans.cluster_centers_
    super_centers = super_centers / np.linalg.norm(super_centers, axis=1, keepdims=True)

    # Map super cluster id -> list of cluster ids
    supercluster_to_clusters = {i: [] for i in range(n_superclusters)}
    for cluster_id, super_label in enumerate(super_labels):
        supercluster_to_clusters[super_label].append(cluster_id)
    
    return supercluster_to_clusters, cluster_to_arxiv


def extract_topic(
    llm,
    arxiv_ids: Optional[List[str]] = None,
    title: Optional[List[str]] = None,
    abstract: Optional[List[str]] = None,
    sample_size: int = 8,
    num_rounds: int = 5,
    cost_per_mil_input_token: float = 0.15,
    cost_per_mil_output_token: float = 0.6
) -> str:
    """
    Extract a consolidated AI topic from a set of papers using OpenAI.
    
    Parameters:
        llm: A ChatOpenAI instance (or similar) to query the LLM.
        arxiv_ids (Optional[List[str]]): List of arXiv IDs to fetch from the database.
            Mutually exclusive with providing title and abstract.
        title (Optional[List[str]]): List of paper titles (if arxiv_ids not provided).
        abstract (Optional[List[str]]): List of paper abstracts (if arxiv_ids not provided).
        sample_size (int): Number of papers to include in each round.
        num_rounds (int): Number of rounds to query the LLM for a topic guess.
        cost_per_mil_input_token (float): Cost per 1,000,000 input tokens.
        cost_per_mil_output_token (float): Cost per 1,000,000 output tokens.
    
    Returns:
        final_topic (str): The consolidated AI topic extracted.
    """
    global accumulated_cost

    if arxiv_ids is not None:
        if title is not None or abstract is not None:
            raise ValueError("Provide either arxiv_ids or title/abstract, not both.")
        conn = return_conn()
        cur = conn.cursor()
        placeholders = ','.join(['%s'] * len(arxiv_ids))
        query = f"SELECT title, abstract FROM paper WHERE arxiv_id IN ({placeholders})"
        cur.execute(query, tuple(arxiv_ids))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        if not rows:
            raise ValueError("No papers found for the provided arxiv_ids.")
        # We only need titles and abstracts.
        titles_list, abstracts_list = zip(*rows)
        titles_list = list(titles_list)
        abstracts_list = list(abstracts_list)
    else:
        if title is None or abstract is None:
            raise ValueError("Either provide arxiv_ids or both title and abstract.")
        if len(title) != len(abstract):
            raise ValueError("Title and abstract lists must have the same length.")
        titles_list = title
        abstracts_list = abstract

    # Combine the paper information (we no longer track arxiv_ids).
    papers = list(zip(titles_list, abstracts_list))
    n_papers = len(papers)
    if n_papers == 0:
        raise ValueError("No papers provided.")

    # Sampling: draw sample_size * num_rounds indices.
    total_needed = sample_size * num_rounds
    if n_papers >= total_needed:
        sample_indices = random.sample(range(n_papers), total_needed)
    else:
        sample_indices = list(range(n_papers))
        random.shuffle(sample_indices)

    # Partition the sampled indices into batches (one batch per round).
    batches = []
    for r in range(num_rounds):
        batch_indices = sample_indices[r * sample_size:(r + 1) * sample_size]
        if batch_indices:  # Only include non-empty batches.
            batch = [papers[i] for i in batch_indices]
            batches.append(batch)

    topic_guesses = []

    # Process each round (batch) and query the LLM.
    for round_idx, batch in enumerate(batches):
        prompt_lines = ["The following are paper titles and abstracts:"]
        for idx, (paper_title, paper_abstract) in enumerate(batch, 1):
            prompt_lines.append(f"{idx}. Title: {paper_title}\nAbstract: {paper_abstract}")
        prompt_lines.append("\nExtract ONE specific AI topic that encapsulates these papers (optimally 2-4 words, max 6)")
        prompt = "\n\n".join(prompt_lines)

        result = get_chat_completion(llm, prompt)
        guess = result["content"].strip()
        input_tokens = result["input_tokens"]
        output_tokens = result["output_tokens"]
        cost_input = (input_tokens / 1e6) * cost_per_mil_input_token
        cost_output = (output_tokens / 1e6) * cost_per_mil_output_token
        cost = cost_input + cost_output
        accumulated_cost += cost

        print(f"Round {round_idx+1}/{len(batches)} - Extracted topic guess: {guess}")
        print(f"  Tokens: input {input_tokens}, output {output_tokens} -> Cost for this call: ${cost:.4f}")
        print(f"  Accumulated cost so far: ${accumulated_cost:.4f}")
        topic_guesses.append(guess)

    # If we only have one round, use the single topic guess directly
    if len(topic_guesses) == 1:
        final_topic = topic_guesses[0]
        # No API call was made, so no additional tokens to count
        input_tokens = 0
        output_tokens = 0
        cost = 0
    else:
        # Consolidate the noisy topic guesses when we have multiple rounds
        final_prompt_lines = ["The following are noisy versions of an AI topic:"]
        for idx, guess in enumerate(topic_guesses, 1):
            final_prompt_lines.append(f"{idx}. {guess}")
        final_prompt_lines.append("\nDetermine the true AI topic (optimally 2-4 words, max 6).")
        final_prompt = "\n\n".join(final_prompt_lines)

        final_result = get_chat_completion(llm, final_prompt)
        final_topic = final_result["content"].strip()
        input_tokens = final_result["input_tokens"]
        output_tokens = final_result["output_tokens"]
        cost_input = (input_tokens / 1e6) * cost_per_mil_input_token
        cost_output = (output_tokens / 1e6) * cost_per_mil_output_token
        cost = cost_input + cost_output
        accumulated_cost += cost

    print("Final topic extraction:")
    print(f"  Final topic: {final_topic}")
    print(f"  Tokens: input {input_tokens}, output {output_tokens} -> Cost for this call: ${cost:.4f}")
    print(f"  Total accumulated cost: ${accumulated_cost:.4f}")

    return final_topic


def extract_cluster_topics(
    cluster_to_arxiv: dict,
    id2title: dict,
    id2abstract: dict,
    llm,
    sample_size: int = 8,
    num_rounds: int = 5,
    cost_per_mil_input_token: float = 0.15,
    cost_per_mil_output_token: float = 0.6
) -> dict:
    """
    For each cluster (keyed by cluster ID in cluster_to_arxiv), extract a topic using extract_topic.
    
    Parameters:
        cluster_to_arxiv (dict): Maps cluster ID to a list of arxiv IDs.
        id2title (dict): Maps arxiv ID to paper title.
        id2abstract (dict): Maps arxiv ID to paper abstract.
        llm: An instance of your LLM.
        sample_size (int): Number of papers to sample per round for topic extraction.
        num_rounds (int): Number of rounds to query the LLM when extracting a topic.
        cost_per_mil_input_token (float): Cost per 1e6 input tokens.
        cost_per_mil_output_token (float): Cost per 1e6 output tokens.
    
    Returns:
        cluster_topics (dict): Maps each cluster ID to its extracted topic.
    """
    cluster_topics = {}
    for cluster_id, arxiv_ids in cluster_to_arxiv.items():
        # Get the titles and abstracts for this cluster using the dictionaries
        titles = [id2title[arxiv_id] for arxiv_id in arxiv_ids]
        abstracts = [id2abstract[arxiv_id] for arxiv_id in arxiv_ids]
        
        print(f"Extracting topic for cluster {cluster_id} with {len(titles)} papers...")
        
        # Call the extract_topic function (which queries the LLM across multiple rounds)
        topic = extract_topic(
            llm=llm,
            title=titles,
            abstract=abstracts,
            sample_size=sample_size,
            num_rounds=num_rounds,
            cost_per_mil_input_token=cost_per_mil_input_token,
            cost_per_mil_output_token=cost_per_mil_output_token
        )
        print(f"Cluster {cluster_id} topic: {topic}\n")
        cluster_topics[cluster_id] = topic
    return cluster_topics



def extract_super_cluster_topics(
    cluster_topics: dict,
    super_to_clusters: dict,
    llm,
) -> dict:
    """
    Using the cluster_topics and super_to_clusters mapping, extract an overarching topic for each super-cluster.
    
    Parameters:
        cluster_topics (dict): Maps each cluster ID to its extracted topic.
        super_to_clusters (dict): Maps each super-cluster ID to a list of cluster IDs.
        llm: An instance of your LLM.
        cost_per_mil_input_token (float): Cost per 1e6 input tokens.
        cost_per_mil_output_token (float): Cost per 1e6 output tokens.
    
    Returns:
        super_cluster_topics (dict): Maps each super-cluster ID to its overarching topic.
    """
    super_cluster_topics = {}
    for super_id, cluster_ids in super_to_clusters.items():
        # Gather the sub-topics from the clusters in this super-cluster.
        sub_topics = [cluster_topics[cid] for cid in cluster_ids if cid in cluster_topics]
        if not sub_topics:
            print(f"No sub-topics found for super-cluster {super_id}. Skipping.")
            continue
        
        # Build a prompt that lists the sub-topics.
        prompt_lines = ["Given these sub-topics:"]
        for idx, sub_topic in enumerate(sub_topics, 1):
            prompt_lines.append(f"{idx}. {sub_topic}")
        # Return a concise phrase (2–4 words, maximum 6 words) that best represents the primary AI domain implied by the subtopics.
        prompt_lines.append("\nExtract the overarching AI topic from the subtopics above. "
                            "Return a concise phrase (1-3 words, max 4) that best represents "
                            "the primary AI domain implied by the subtopics. Example outputs "
                            "include `Computer Vision`, `Natural Language Processing`, "
                            "`Reinforcement Learning`, `Robotics`, `Medical AI`.")
        final_prompt = "\n\n".join(prompt_lines)

        # Query the LLM to extract the overarching topic.
        final_result = get_chat_completion(llm, final_prompt)
        overarching_topic = final_result["content"].strip()
        print(f"Super-cluster {super_id} overarching topic: {overarching_topic}\n")
        super_cluster_topics[super_id] = overarching_topic
    return super_cluster_topics

def dedupe_super_cluster_topics(
    super_cluster_topics: dict,
    super_to_clusters: dict
) -> tuple:
    """
    De-duplicate super clusters that have the exact same topic.
    
    Parameters:
        super_cluster_topics (dict): Maps each super-cluster ID to its topic.
        super_to_clusters (dict): Maps each super-cluster ID to a list of cluster IDs.
    
    Returns:
        tuple: (deduped_super_cluster_topics, deduped_super_to_clusters)
            - deduped_super_cluster_topics: Maps each super-cluster ID to its topic (with duplicates removed).
            - deduped_super_to_clusters: Maps each super-cluster ID to a list of cluster IDs (with merged clusters).
    """
    # Group super clusters by their topics
    topic_to_super_ids = {}
    for super_id, topic in super_cluster_topics.items():
        if topic not in topic_to_super_ids:
            topic_to_super_ids[topic] = []
        topic_to_super_ids[topic].append(super_id)
    
    # Create new mappings without duplicates
    deduped_super_cluster_topics = {}
    deduped_super_to_clusters = {}
    
    for topic, super_ids in topic_to_super_ids.items():
        # Keep the first super_id and merge others into it
        primary_super_id = super_ids[0]
        deduped_super_cluster_topics[primary_super_id] = topic
        
        # Merge the cluster lists
        merged_clusters = []
        for super_id in super_ids:
            merged_clusters.extend(super_to_clusters[super_id])
        deduped_super_to_clusters[primary_super_id] = merged_clusters
    
    print(f"Reduced from {len(super_cluster_topics)} to {len(deduped_super_cluster_topics)} super clusters by deduplication.")
    
    return deduped_super_cluster_topics, deduped_super_to_clusters


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(description='Two-tiered k-means clustering for paper topic extraction')
        parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing documents')
        parser.add_argument('--n-clusters', type=int, default=100, help='Number of clusters for first k-means')
        parser.add_argument('--n-superclusters', type=int, default=12, help='Number of super clusters for second k-means')
        parser.add_argument('--random-state', type=int, default=42, help='Random seed for reproducibility')
        parser.add_argument('--sample-size', type=int, default=8, help='Number of papers to sample per round')
        parser.add_argument('--num-rounds', type=int, default=1, help='Number of rounds for topic extraction')
        parser.add_argument('--openai-api-key', type=str, help='OpenAI API key')
        return parser.parse_args()

    args = parse_args()

    # Init ChatOpenAI
    if args.openai_api_key and args.openai_api_key != "":
        llm = ChatOpenAI(api_key=args.openai_api_key, max_completion_tokens=15)
    else:
        llm = ChatOpenAI(max_completion_tokens=15)  # should set env variable
    
    healthy_connection = get_chat_completion(llm, "Healthy connection? Reply `yes` or `no`.")
    if healthy_connection['content'].lower().strip() != "yes":
        raise ValueError("LLM connection is not healthy.")

    # extract embeddings
    embeddings, arxiv_ids, id2title, id2abstract = build_embeddings(batch_size=args.batch_size)

    # supercluster IDs to cluster IDs, cluster IDs to arxiv IDs
    supercluster_to_clusters, cluster_to_arxiv = cluster_arxiv_embeddings(
        embeddings=embeddings,
        arxiv_ids=arxiv_ids,
        n_clusters=args.n_clusters,
        n_superclusters=args.n_superclusters,
        random_state=args.random_state
    )

    # Display the titles of the first 5 arxiv_ids in first 5 clusters.
    for cluster_id in range(5):
        print(f"\nCluster {cluster_id}:")
        for arxiv_id in cluster_to_arxiv[cluster_id][:5]:
            print(f"{arxiv_id}: {id2title[arxiv_id]}")

    # Get the semantic meanings of the clusters

    cluster_topics = extract_cluster_topics(
        cluster_to_arxiv,
        id2title,
        id2abstract,
        llm,
        sample_size=args.sample_size,
        num_rounds=args.num_rounds,
    )

    super_cluster_topics = extract_super_cluster_topics(
        cluster_topics,
        supercluster_to_clusters,
        llm
    )

    deduped_super_cluster_topics, deduped_supercluster_to_clusters = dedupe_super_cluster_topics(
        super_cluster_topics,
        supercluster_to_clusters
    )

    # Create output dictionary
    output_data = []

    # Map each arxiv_id to its cluster_id and supercluster_id
    for supercluster_id, cluster_ids in deduped_supercluster_to_clusters.items():
        supercluster_topic = deduped_super_cluster_topics[supercluster_id]
        
        for cluster_id in cluster_ids:
            if cluster_id in cluster_topics:
                cluster_topic = cluster_topics[cluster_id]
                
                for arxiv_id in cluster_to_arxiv[cluster_id]:
                    output_data.append({
                        "arxiv_id": arxiv_id,
                        "supercluster_topic": supercluster_topic,
                        "cluster_topic": cluster_topic
                    })

    # Write to JSON file
    with open(TOPICS_PATH, 'w') as f:
        for item in output_data:
            f.write(json.dumps(item) + '\n')

    print(f"Results written to {TOPICS_PATH}")
    print(f"Total papers processed: {len(output_data)}")
    print(f"Total superclusters: {len(deduped_supercluster_to_clusters)}")
    print(f"Total clusters: {len(cluster_topics)}")
