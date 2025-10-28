import torch
from typing import Iterable, List, Tuple
from collections import Counter
import os

from data_pipe import iter_wiki_sentences

import spacy 
nlp = spacy.load("en_core_web_sm")

AUX = {"do","does","did","am","is","are","was","were","be","been","being",
       "have","has","had","will","would","shall","should","can","could",
       "may","might","must"}
INTENS = {"really","very","quite","so","too","extremely","fairly","pretty",
          "rather","somewhat","kinda","sorta","at","all"}
POS_PAIR = {("PROPN","PROPN"),("ADJ","NOUN"),("NOUN","NOUN"),("PROPN","NOUN"),("NOUN","PROPN")}

negate = {"no", "not", "never"}
BAD_PART = {"not", "to"}
common_words = [
    "the","a","an","of","and","to","in","on","for","with","at","by","from",
    "about","as","is","was","are","were","be","been","being",
    "do","does","did","have","has","had",
    "that","this","these","those","it","its",
    "i","you","he","she","they","we","me","him","her","them","us",
    "my","your","his","their","our",
    "but","or","so","if","then",
    "there","here","when","where","what","which","who","whom",
    "gonna","gotta","wanna","lemme","gimme","imma","outta","kinda"
]


def build_vocab(token_iter: Iterable[str],
                min_count=None, specials: List[str]=None):
    if specials is None:
        specials = ["<unk>"]

    counter = Counter()
    total = 0
    for sents in token_iter:
        print(sents)
        for tok in sents:
            counter[tok] += 1
            total += 1


    vocab = [w for w, i in counter.items() if i >= min_count]
    vocab = sorted(vocab, key=lambda w:-counter[w])


    id2word = list(specials) + [w for w in vocab]
    word2id = {w:i for i, w in enumerate(id2word)}

    mask = torch.zeros(len(id2word), dtype=torch.bool)
    for idx, word in enumerate(id2word):
        doc = nlp(word)
        if doc[0].pos_ == "ADJ":
            mask[idx] = True

    counts = torch.tensor([counter[w] for w in id2word], dtype=torch.long)
    
    return word2id, id2word, counts, total, mask

def compute_keep_probs(counts: torch.Tensor=None,
                    total_tokens: int=None, t=1e-5, mask: torch.BoolTensor=None) -> torch.Tensor:
    device = counts.device

    count_f = counts.to(torch.float32)
    t = torch.tensor(t, dtype=torch.float32, device=device)
    f = count_f / float(total_tokens)
    p = torch.ones_like(f)

    nz = f > 0
    
    p[nz] = (torch.sqrt(f[nz]/t) + 1) * (t / f[nz])
    p[mask] = torch.minimum(p[mask], torch.tensor(0.5, device=device))
    return torch.clamp(p, 0, 1).to(torch.float32)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "..", "..", "..", "data")
    save_path = os.path.join(data_dir, "vocab.pt")

    train_iter = iter_wiki_sentences("train")

    
    word2id, id2word, counts, count, mask = build_vocab(train_iter, min_count=22, specials=["<unk>"])
    print(len(counts))
    print(len(word2id))
    keep_probs = compute_keep_probs(counts, count, t=5e-6)
    obj = {"word2id": word2id,
           "id2word": id2word,
           "counts" : counts,
           "count"  : count,
           "mask" : mask,
        "keep_probs": keep_probs}
    
    
    torch.save(obj, save_path)
    print(f"saved vocab, length:{len(word2id)}")
    
