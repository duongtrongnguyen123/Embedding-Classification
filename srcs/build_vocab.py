import torch
from datasets import load_dataset
from typing import Iterable, List
from collections import Counter

import os

from data_pipe import (
    tokenize, iter_wiki_tokens,
    load_data
)

def build_vocab(token_iter: Iterable[str],
                min_count=5, max_vocab_size=None,
                specials: List[str]=None):
    if specials is None:
        specials = ["<unk>"]

    counter = Counter()
    total = 0
    for tok in token_iter:
        total += 1
        counter[tok] += 1

    vocab = [w for w, i in counter.items() if i >= min_count]
    vocab = sorted(vocab, key=lambda w:-counter[w])

    if max_vocab_size is not None:
        cap = max(0, max_vocab_size - len(specials))
        vocab = vocab[:cap]

    id2word = list(specials) + [w for w in vocab]
    word2id = {w:i for i, w in enumerate(id2word)}

    counts = torch.tensor([counter[w] for w in id2word], dtype=torch.long)
    return word2id, id2word, counts, total

def compute_keep_probs(counts: torch.Tensor=None,
                    total_tokens: int=None, t=1e-4) -> torch.Tensor:
    device = counts.device

    count_f = counts.to(torch.float32)
    t = torch.tensor(t, dtype=torch.float32, device=device)
    f = count_f / float(total_tokens)
    p = torch.ones_like(f)

    nz = f > 0
    
    p[nz] = (torch.sqrt(f[nz]/t) + 1) * (t / f[nz])
    return torch.clamp(p, 0, 1).to(torch.float32)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")
    

    train_iter = iter_wiki_tokens("train")
    valid_iter = iter_wiki_tokens("validation")
    test_iter = iter_wiki_tokens("test")

    word2id, id2word, counts, count = build_vocab(train_iter, min_count=10, max_vocab_size=None)
    print(len(word2id))
    keep_probs = compute_keep_probs(counts, count)
    obj = {"word2id": word2id,
           "id2word": id2word,
           "counts" : counts,
           "count"  : count,
        "keep_probs": keep_probs}
    
    save_path = os.path.join(data_dir, "vocab.pt")
    
    torch.save(obj, save_path)
    print(f"saved vocab, length:{len(word2id)}")