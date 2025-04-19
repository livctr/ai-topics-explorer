import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Dict

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
    json_path: str,
    model_name: str = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct',
    batch_size: int = 16,
    device: torch.device = None,
    max_length: int = 8192
) -> Dict[str, torch.Tensor]:
    """
    Returns a dict mapping paper_id → embedding (CPU Tensor).
    Embeddings are L2-normalized pooled token representations.
    """
    # pick device
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    # load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    # load papers
    with open(json_path, 'r') as f:
        papers_dict = json.load(f)

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
        for i in range(0, len(papers), batch_size):
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
                embeddings[pid] = pooled[j]

    return embeddings


if __name__ == '__main__':
    import pdb ; pdb.set_trace()
    embs = get_papers_embeddings('papers.json', batch_size=32)
    print(embs)