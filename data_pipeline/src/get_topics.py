import random
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Dict

from typing import Any, Dict, List, Tuple
import numpy as np
from k_means_constrained import KMeansConstrained

from tqdm import tqdm

import os
from transformers import AutoModelForCausalLM, AutoTokenizer


def last_token_pool(last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
    # as you already have it…
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]


def get_papers_embeddings(
    papers_dict,
    embds_model_name: str,
    batch_size: int = 16,
    max_length: int = 8192
) -> Dict[str, torch.Tensor]:
    """
    Returns a dict mapping paper_id → embedding (CPU Tensor).
    Embeddings are L2-normalized pooled token representations.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(embds_model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(embds_model_name, trust_remote_code=True).to(device)
    model.eval()

    # build list of (id, text, token_length)
    papers = []
    for pid, entry in papers_dict.items():
        text = entry['title'].strip()
        if entry.get('abstract'):
            text += '. ' + entry['abstract'].strip()
        # quick token count
        tok_len = len(tokenizer.tokenize(text))
        papers.append((pid, text, tok_len))

    # sort by length descending
    papers.sort(key=lambda x: x[2], reverse=True)

    embeddings: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for i in tqdm(range(0, len(papers), batch_size), desc="Generating embeddings..."):
            batch = papers[i:i+batch_size]
            texts = [t for (_, t, _) in batch]
            enc = tokenizer(
                texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(device)
            out = model(**enc)
            pooled = last_token_pool(out.last_hidden_state, enc['attention_mask'])
            pooled = F.normalize(pooled, p=2, dim=1).cpu()
            for j, (pid, _, _) in enumerate(batch):
                embeddings[pid] = pooled[j].cpu().numpy()

    del model, tokenizer
    torch.cuda.empty_cache()
    return embeddings


def cluster_embeddings_two_level_balanced(
    embs: Dict[Any, np.ndarray],
    n_clusters: int = 144,
    n_superclusters: int = 12,
    tol: float = 0.2,
    random_state: int = 42
) -> Tuple[Dict[int, List[int]], Dict[int, List[Any]]]:
    """
    Two-level balanced clustering using KMeansConstrained.

    Args:
        embs:            Mapping from ID -> 1D numpy embedding (shape (d,)).
        n_clusters:      Number of first-level clusters.
        n_superclusters: Number of second-level clusters on cluster centers.
        tol:             Fractional tolerance for cluster size balancing (e.g. 0.1 = ±10%).
        random_state:    Seed for reproducibility.

    Returns:
        super_to_clusters: Dict mapping supercluster ID -> list of first-level cluster IDs.
        cluster_to_ids:    Dict mapping first-level cluster ID -> list of original IDs.
    """
    # 1) Prepare data
    ids = list(embs.keys())
    X = np.vstack([embs[_id] for _id in ids])  # shape (N, d)
    N = len(ids)

    # 2) First-level balanced k-means
    ideal_size = N / n_clusters
    size_min = int((1 - tol) * ideal_size)
    size_max = int((1 + tol) * ideal_size)

    km1 = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=size_min,
        size_max=size_max,
        random_state=random_state
    )

    labels1 = km1.fit_predict(X)
    cluster_centers = km1.cluster_centers_
    # normalize centers on sphere
    cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)

    # map cluster -> IDs
    cluster_to_ids: Dict[int, List[Any]] = {i: [] for i in range(n_clusters)}
    for idx, lbl in enumerate(labels1):
        cluster_to_ids[lbl].append(ids[idx])

    # 3) Second-level balanced k-means on cluster centers
    M = n_clusters
    ideal2 = M / n_superclusters
    size_min2 = int((1 - tol) * ideal2)
    size_max2 = int((1 + tol) * ideal2)

    km2 = KMeansConstrained(
        n_clusters=n_superclusters,
        size_min=size_min2,
        size_max=size_max2,
        random_state=random_state
    )

    labels2 = km2.fit_predict(cluster_centers)

    # map supercluster -> cluster IDs
    super_to_clusters: Dict[int, List[int]] = {i: [] for i in range(n_superclusters)}
    for cluster_id, super_lbl in enumerate(labels2):
        super_to_clusters[super_lbl].append(cluster_id)

    return super_to_clusters, cluster_to_ids


def get_cluster_topics(
    super_to_clusters: Dict[int, List[int]],
    cluster_to_ids: Dict[int, List[Any]],
    papers_dict: Dict[str, Dict[str, str]],
    cluster_naming_model: str,
    sample_size: int = 8,
    num_rounds: int = 1,
    batch_size: int = 8,
) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Generates descriptive topics for clusters and superclusters using a local HF LLM.

    Args:
        super_to_clusters: Mapping from supercluster ID -> list of first-level cluster IDs.
        cluster_to_ids:    Mapping from cluster ID -> list of paper IDs.
        papers_dict:       Dict mapping paper ID -> {'title': ..., 'abstract': ...}.
        model:             A pretrained HuggingFace AutoModelForCausalLM instance.
        tokenizer:         Corresponding AutoTokenizer.
        device:            Torch device (e.g. 'cuda' or 'cpu').
        sample_size:       Max number of papers to sample per round.
        num_rounds:        Number of random sampling rounds per cluster.

    Returns:
        super_topics: Mapping supercluster ID -> descriptive string.
        cluster_topics: Mapping cluster ID -> descriptive string.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained(
        cluster_naming_model,
        torch_dtype="auto",
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cluster_naming_model, padding_side="left")

    def generate_from_prompts(prompts: List[str]) -> List[str]:
        message_batch = [
            [{"role": "user", "content": prompt}]
            for prompt in prompts
        ]
        
        responses = []

        for i in range(0, len(message_batch), batch_size):
            message_batch_i = message_batch[i:i + batch_size]
            text_batch = tokenizer.apply_chat_template(
                message_batch_i,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs_batch = tokenizer(text_batch, return_tensors="pt", padding=True).to(model.device)

            generated_ids_batch = model.generate(
                **model_inputs_batch,
                max_new_tokens=32,
            )
            generated_ids_batch = generated_ids_batch[:, model_inputs_batch.input_ids.shape[1]:]
            response_batch = tokenizer.batch_decode(generated_ids_batch, skip_special_tokens=True)
            responses.extend(response_batch)
        return responses
    

    def build_paper_prompt(papers: List[Tuple[str, str]]) -> str:
        lines = []
        for i, (title, abstract) in enumerate(papers, 1):
            if len(abstract) > 0:
                lines.append(f"{i}. Title: {title}\nAbstract: {abstract}")
            else:
                lines.append(f"{i}. Title: {title}")
        body = "\n\n".join(lines)
        return f"{body}\n\nExtract ONE specific AI topic that encapsulates these papers (optimally 2-4 words, max 6)"

    def build_aggregate_prompt(guesses: List[str]) -> str:
        lines = [f"The following are noisy versions of ONE AI topic:"]
        for i, g in enumerate(guesses, 1):
            lines.append(f"{i}. {g}")
        body = "\n".join(lines)
        return f"{body}\n\nDetermine ONE true AI topic (optimally 2-4 words, max 6)."

    # 1) cluster topics
    cluster_topics: Dict[int, str] = {}
    for cid, ids in tqdm(cluster_to_ids.items(), desc="Generating cluster topics..."):
        if not ids:
            cluster_topics[cid] = ''
            continue
        prompts: List[str] = []
        for _ in range(num_rounds):
            sample = ids if len(ids) <= sample_size else random.sample(ids, sample_size)
            papers = [(papers_dict[_id]['title'], papers_dict[_id]['abstract']) for _id in sample]
            prompt = build_paper_prompt(papers)
            prompts.append(prompt)
        guesses = generate_from_prompts(prompts)
        if num_rounds == 1:
            cluster_topics[cid] = guesses[0]
        else:
            agg_prompt = build_aggregate_prompt(guesses)
            cluster_topics[cid] = generate_from_prompts([agg_prompt])[0]

    # 2) supercluster topics with multi-round logic
    super_topics: Dict[int, str] = {}
    for scid, cids in tqdm(super_to_clusters.items(), desc="Generating supercluster topics..."):
        if not cids:
            super_topics[scid] = ''
            continue
        prompts: List[str] = []
        for _ in range(num_rounds):
            sample = cids if len(cids) <= sample_size else random.sample(cids, sample_size)
            papers = [(f"Cluster {cid}", cluster_topics[cid]) for cid in sample if cluster_topics.get(cid)]
            prompt = build_paper_prompt(papers)
            prompts.append(prompt)
        guesses = generate_from_prompts(prompts)
        if num_rounds == 1:
            super_topics[scid] = guesses[0]
        else:
            agg_prompt = build_aggregate_prompt(guesses)
            super_topics[scid] = generate_from_prompts([agg_prompt])[0]

    return super_topics, cluster_topics
