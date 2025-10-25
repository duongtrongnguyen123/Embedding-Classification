import pyximport
pyximport.install(language_level=3)


import torch
import numpy as np
from encode_corpus import encode_corpus
from fast_count import iter_sentences
import os


negate = {"not", "no", "never"}

AUX = {"do","does","did","am","is","are","was","were","be","been","being",
       "have","has","had","will","would","shall","should","can","could",
       "may","might","must"}
INTENS = {"really","very","quite","so","too","extremely","fairly",
                       "pretty","rather","somewhat","kinda","sorta","at","all"}



pairs_tokens = [
    "machine","learning","is","nigga","popular","in","many","fields",".",
    "deep","learning","is","a","subfield","of","machine","learning",".",
    "data","science","not","really","quite", "often","uses","machine","learning","methods","here",".",
    "artificial","intelligence","includes","machine","learning","and","deep","learning",".",
    "neural","networks","are","the","basis","of","deep","learning",".",
    "python","is","commonly","used","for","machine","learning","and","data","science",".",
    "machine","learning","algorithms","can","be","supervised","or","unsupervised",".",
    "deep","learning","can","train","large","neural","networks","with","big","data",".",
    "data","science","combines","statistics","and","machine","learning",".",
    "machine","learning","applications","include","vision","speech","and","language",".",
    "deep","learning","applications","include","image","recognition","and","translation",".",
    "many","companies","invest","in","data","science","and","machine","learning",".",
    "researchers","publish","papers","about","deep","learning","and","machine","learning",".",
    "students","study","data","science","and","practice","machine","learning",".",
    "machine","learning","is","used","in","recommendation","systems",".",
    "deep","learning","is","used","in","autonomous","vehicles",".",
    "data","science","is","important","for","business",".",
    "machine","learning","models","require","training","data",".",
    "deep","learning","models","require","a","lot","of","data",".",
    "machine","learning","and","data","science","are","growing",".",
    "machine","learning","and","deep","learning","are","closely","related",".",
    "data","science","teams","work","with","machine","learning",".",
    "deep","learning","research","advances","every","year",".",
    "machine","learning","helps","solve","real","world","problems",".",
    "data","science","helps","companies","make","better","decisions",".",
    "deep","learning","helps","improve","speech","recognition",".",
    "machine","learning","is","used","for","fraud","detection",".",
    "data","science","is","used","for","customer","analytics",".",
    "deep","learning","is","not","used","for","medical","diagnosis",".",
    "machine","learning","requires","features","and","labels",".",
    "deep","learning","requires","powerful","hardware",".",
    "data","science","requires","data","collection",".",
    "machine","learning","is","at","the","core","of","modern","ai",".",
    "deep","learning","is","a","key","part","of","modern","ai",".",
    "data","science","is","a","key","part","of","modern","business",".",
    "machine","learning","and","deep","learning","drive","innovation",".",
    "students","learn","machine","learning","and","data","science",".",
    "researchers","explore","deep","learning","and","machine","learning",".",
    "companies","hire","data","science","teams",".",
    "machine","learning","is","taught","in","universities",".",
    "deep","learning","is","taught","in","advanced","courses",".",
    "data","science","is","taught","in","many","programs","."
]


def get_aux_intens_id(o_word2id, a, b):
    res = set()
    for s in a:
        id = o_word2id.get(s, None)
        if id is not None:
            res.add(id)
    for s in b:
        id = o_word2id.get(s, None)
        if id is not None:
            res.add(id)
        
    return res

def get_negate_id(o_word2id, negate):
    res = set()
    for s in negate:
        id = o_word2id.get(s, None)
        if id is not None:
            res.add(id)

    return res


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    saved_corpus_dir = os.path.join(base_dir, "..", "data", "test", "o_corpus.bin")
    to_save_corpus_dir = os.path.join(base_dir, "..", "data", "test", "n_corpus.bin")
    
    vocab_dir = os.path.join(base_dir, "..", "data", "test", "vocab.pt")
    vocab = torch.load(vocab_dir)
    o_word2id = vocab["o_word2id"]
    old2new = vocab["old2new"]
    old2new_for_pair = vocab["old2new_for_pair"]
    skip_id = get_aux_intens_id(o_word2id, AUX, INTENS)
    negate_id = get_negate_id(o_word2id, negate)
    id2word = vocab["id2word"] 

    encode_corpus(old2new, negate_id, skip_id, old2new_for_pair, saved_corpus_dir, to_save_corpus_dir, id2word)

    


