import torch
from typing import Iterable, List, Tuple
from collections import Counter
import math
import os

from data_pipe import iter_wiki_sentences_for_compute_pairs, iter_wiki_sentences

import spacy 
nlp = spacy.load("en_core_web_sm")

AUX = {"do","does","did","am","is","are","was","were","be","been","being",
       "have","has","had","will","would","shall","should","can","could",
       "may","might","must"}
INTENS = {"really","very","quite","so","too","extremely","fairly","pretty",
          "rather","somewhat","kinda","sorta","at","all"}

def build_stas(token_iter: Iterable[str], min_count_bi: int=9,
               min_thres: int=2, top_k: int=3000):
    unigram = Counter()
    bigram = Counter()
    total_u = 0
    total_bi = 0
    unigram["<unk>"] = 0
    for sents in token_iter:
        prev_tok = None
        for tok in sents:
            if prev_tok:
                if (nlp(prev_tok)[0].pos_, nlp(tok)[0].pos_) in {("PROPN","PROPN"),("ADJ","NOUN"),("NOUN","NOUN"),("VERB","PART"),("PROPN","NOUN"),("NOUN","PROPN")}:
                    bigram[(prev_tok, tok)] += 1
                    prev_tok = None
                elif prev_tok in ("no", "never", "not"): 
                    if tok in AUX or tok in INTENS:
                        continue
                    bigram[(prev_tok, tok)] += 1
                    prev_tok = None
            unigram[tok] += 1
            total_u += 1
            prev_tok = tok
    
    scored: list[tuple[float, tuple[str, str], int]] = []
    for (c1, c2), count_bi in bigram.item():
        if count_bi < min_count_bi:
            continue

        p12 = count_bi / total_bi
        p1 = unigram[c1] / total_u
        p2 = unigram[c2] / total_u
        de = p1 * p2

        if de <= 0: 
            continue

        ppmi = math.log2(p12 / de)

        if ppmi < min_thres:
            continue
        scored.append((ppmi, (c1, c2), count_bi))

    scored.sort(key=lambda x:(x[0], x[2 ]))

    scored = scored[:top_k]
    
    top_pairs: set[tuple[str,str]] = {p for _,p,_ in scored}
    return unigram, total_u, bigram, total_bi, top_pairs
    


def build_vocab(token_iter: Iterable[str],
                min_count=None, unigram: Counter=None, 
                bigram: Counter=None, total_u: int=None, 
                total_bi: int=None, specials: List[str]=None):
    if specials is None:
        specials = ["<unk>"]

    counts = Counter()
    for sents in token_iter:




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
    data_dir = os.path.join(base_dir, "..", "data")
    

    train_iter = iter_wiki_sentences("train")
    #valid_iter = iter_wiki_tokens("validation")
    #test_iter = iter_wiki_tokens("test")

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
    
    save_path = os.path.join(data_dir, "vocab.pt")
    
    torch.save(obj, save_path)
    print(f"saved vocab, length:{len(word2id)}")