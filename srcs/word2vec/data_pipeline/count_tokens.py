import pyximport
pyximport.install(language_level=3)

from _count_fast import first_pass, save_valid_encode, build_vocab
from review_dataset_iter import train_iter_review_sentences, valid_iter_review_sentences
import torch
import numpy as np
import spacy
import os
nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

POS_PAIR = {("ADJ","NOUN"), ("NOUN","NOUN"), ("PROPN","PROPN")}

negate = {"no", "not", "never"}
merge_with_negate = {"ADJ", "VERB", "ADV"}
BAD_PART = {"not", "to"}
AUX = {"do","does","did","am","is","are","was","were","be","been","being",
       "have","has","had","will","would","shall","should","can","could",
       "may","might","must"}
INTENS = {"really","very","quite","so","too","extremely","fairly",
                       "pretty","rather","somewhat","kinda","sorta","at","all"}




common_words = [
    "the","a","an","of","and","to","in","on","for","with","at","by","from",
    "about","as","is","was","are","were","be","been","being",
    "do","does","did","have","has","had",
    "that","this","these","those","it","its",
    "i","you","he","she","they","we","me","him","her","them","us",
    "my","your","his","their","our",
    "but","or","so","if","then",
    "there","here","when","where","what","which","who","whom",
    "gonna","gotta","wanna","lemme","gimme","imma","outta","kinda",
    "thing", "everyone", "everybody", "aok_aok", "vg_vg", "soso"
]

enhance = [
"excellent","outstanding","masterpiece","brilliant","moving","gripping","heartfelt",
"charming","delightful","hilarious","clever","smart","entertaining","engaging",
"compelling","impressive","top-notch","superb","well-acted","well-written","must-watch",
"mustsee","rewatchable","worth-watching","underrated","believable","nuanced","satisfying",
"awful","terrible","horrible","dreadful","boring","dull","uneven","messy","incoherent",
"cliché","cliched","cringe","cringey","wooden","flat","shallow","pretentious",
"disappointing","forgettable","overrated","underwritten","tedious","unfunny","predictable",
"derivative","waste","pointless"
]


def pair_id_decode(id):
    return (int(id>>32), int(id & 0XFFFFFFFF))

def get_skip_id(o_word2id, a, b):
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

def get_negate_id(o_word2id, a):
    res = set()
    for s in a:
        id = o_word2id.get(s, None)
        if id is not None:
            res.add(id)

    return res

def vocab_pos(id2word, batch_size=32768, group_size=16):
    pos_arr = []
    with nlp.select_pipes(disable=["parser", "ner"]):
        for doc in nlp.pipe(id2word, batch_size=batch_size):
            pos_arr.append(doc[0].pos_ if doc else "X")
    return pos_arr

def build_mask(word2id, pos_arr):
    adj_arr = [True if pos=="ADJ" else False for pos in pos_arr]
    adv_arr = [True if pos=="ADV" else False for pos in pos_arr]
    common_id = [word2id.get(w, None) for w in common_words]
    
    adj_ten = torch.tensor(adj_arr, dtype=torch.bool)
    adv_ten = torch.tensor(adv_arr, dtype=torch.bool)
    common_ten = torch.zeros(len(word2id), dtype=torch.bool)

    for w in common_id:
        if w is not None:
            common_ten[w] = True
    return adj_ten, adv_ten, common_ten
 


def compute_keep_probs(counts: torch.Tensor=None, t=1e-5, adj_mask: torch.BoolTensor=None, adv_mask: torch.BoolTensor=None, common_mask: torch.BoolTensor=None) -> torch.Tensor:
    device = counts.device

    t = torch.tensor(t, dtype=torch.float32, device=device)
    
    count_f = counts.to(torch.float32)
    total_tokens = count_f.sum()
    f = count_f / float(total_tokens)
    p = torch.zeros_like(f)

    nz = f > 0
    
    p[nz] = (torch.sqrt(f[nz]/t) + 1) * (t / f[nz])
    p[adj_mask] = torch.maximum(p[adj_mask], torch.tensor( 0.45, device=device))
    p[adv_mask] = torch.maximum(p[adv_mask], torch.tensor(0.33, device=device))        
    p[common_mask] = torch.minimum(p[common_mask], torch.tensor(0.001, device=device))

    return torch.clamp(p, 0, 1).to(torch.float32)


def enhnce(word2id, counts):
    for w in enhance:
        id = word2id.get(w, None)
        if id is not None:
            counts[id] += 200
    

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    to_save_vocab_dir = os.path.join(base_dir, "..", "..", "..", "data", "vocab.pt")
    to_save_train_corpus_dir = os.path.join(base_dir, "..", "..", "..", "data", "o_train_corpus.bin")
    to_save_valid_corpus_dir = os.path.join(base_dir, "..", "..", "..", "data", "o_valid_corpus.bin")


    train_iter = train_iter_review_sentences()
    valid_iter = valid_iter_review_sentences()

    print("counting unigram...")
    unigram, top_pairs, o_id2word, o_word2id = first_pass(train_iter, top_k=10000, min_pair_count=6, to_save_path=to_save_train_corpus_dir)
    print("save valid encode...")
    save_valid_encode(valid_iter, o_word2id, to_save_path=to_save_valid_corpus_dir)
    print(len(o_id2word))

    print("converting new vocab...") 
    old2new, word2id, id2word, counts = build_vocab(unigram, o_id2word, o_word2id, min_count=25)
    
    print(len(id2word))

    print("merge with top pair")
    pos_arr = vocab_pos(id2word) 

    old2new_for_pair: dict[int, int] = {}
    counts_for_pair_list = []
    pos_arr_for_pair = []
    for pair_id, count in top_pairs:
        old_id1, old_id2 = pair_id_decode(pair_id)
        new_id1 = old2new[old_id1]
        new_id2 = old2new[old_id2]
        if new_id1 == -1 or new_id2 == -1:
            continue

        a = (pos_arr[new_id1], pos_arr[new_id2]) in POS_PAIR 
        b = (id2word[new_id1] in negate and pos_arr[new_id2] in  merge_with_negate)  
        c = (pos_arr[new_id1] == "VERB" and pos_arr[new_id2] == "PART" and id2word[new_id2] not in BAD_PART)
        if a:
            pos_arr_for_pair.append("NOUN")
        elif b:
            pos_arr_for_pair.append("ADV")
        elif c:
            pos_arr_for_pair.append("VERB")
        else:
            continue

        merge_word = id2word[new_id1] + "_" + id2word[new_id2]
        idx = len(id2word)
        id2word.append(merge_word)
        word2id[merge_word] = idx
        old2new_for_pair[pair_id] = idx
        counts_for_pair_list.append(count + 2)                     #laplace smooth


    pos_arr.extend(pos_arr_for_pair)                           #merge pos with pair
    
    counts = torch.from_numpy(counts)

    counts_for_pair = torch.tensor(
    counts_for_pair_list, 
    dtype=counts.dtype, 
    device=counts.device)

    counts = torch.cat([counts, counts_for_pair], dim=0)


    adj_mask, adv_mask, common_mask = build_mask(word2id, pos_arr)

    print("compute keep probs")

    keep_probs = compute_keep_probs(counts, t=6e-6, adj_mask=adj_mask, adv_mask=adv_mask, common_mask=common_mask)
    
    enhnce(word2id, counts)

    skip_id = get_skip_id(o_word2id, AUX, INTENS)
    negate_id = get_negate_id(o_word2id, negate)

    print(keep_probs[word2id["happy"]])
    print(counts[word2id["abd"]])
    print(counts[word2id["huh"]])

    bundle = { 
        "o_word2id": o_word2id,
        "o_id2word": o_id2word,
        "old2new"  : old2new,                   # np.array
        "old2new_for_pair": old2new_for_pair, # dict
        "word2id"  : word2id,
        "id2word"  : id2word,
        "counts"   : counts,                     # tensor
        "keep_probs": keep_probs,             # tensor
        "skip_id"  : skip_id,                 # old id
        "negate_id": negate_id,               # old id 
        "pos_arr"  : pos_arr                  # new id
    }
    torch.save(bundle, to_save_vocab_dir)

