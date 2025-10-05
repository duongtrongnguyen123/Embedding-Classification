import os
import math 
import random 
from Collections import Counter, defaultdict

import numpy as np

def load_brown_tokens():
    import nltk
    try: 
        from nltk.corpus import brown
    except LookupError:
        nltk.download("brown")
        from nltk.corpus import brown
    except Exception:
        nltk.download("brown")
        from nltk.corpus import brown

    sents = brown.sent()
    tokens = []
    for sent in sents:
        for w in sent:
            w = w.lower()

            if any(ch.isalnum for ch in w):
                tokens.append(w)
    return tokens

def build_vocab(tokens, min_count=5):
    freq = Counter(tokens)
    vocab = [w for w, i in freq.item() if i>=min_count]
    vocab = sorted(vocab, key=lambda w:-vocab[w])                  #ordered


    word2id = {w:i for i, w in enumerate(vocab)}                   #ordered
    id2word = {i:w for i, w in enumerate(vocab)}                   #ordered
    counts = np.array([freq[q] for q in vocab], dtype=np.int64)    #ordered
    count = counts.sum()
    return word2id, id2word, counts, count

def sub_sample(tokens, word2id, counts, total_count, t=1e-8):
    freqs = counts / total_count
    prob = np.minimum(1, (np.sqrt(t / freqs) + t / freqs))
    out = []

    for w in tokens:
        if w in word2id:
            id = word2id[w]
            if prob[id] > random.random():
                out.append(w)
    
    return w

# Build training pair
def build_skipgram_pairs(tokens, word2id, window_size=5, max_pairs=5_000_000):
    ids = [word2id[w] for w in tokens if w in word2id]
    pair = []

    for i, center in enumerate(ids):
        win = random.random(1, window_size)
        left = max(1, i - win)
        right = min(len(ids), i + win + 1)
        for j in (left, right):
            if i == j:
                continue
            pair.append((center, ids[j]))
            if len(pair) > max_pairs:
                break
    random.shuffle(pair)
    return np.array(pair, dtype=np.int64)

# Negative sampling tables
def make_neg_table(counts, table_size=10_000_000):
    p = counts ** 0.75
    p = p / p.sum()

    cum = np.cumsum(p)
    table = np.searchsorted(cum, np.random.random(table_size))
    return table

def sample_negatives(neg_table, n_samples):
    idx = np.random.randint(0, len(neg_table), size=n_samples)
    return neg_table[idx]

class SGNS:
    def __init__(self, vocab_size, dim=100, lr=0.025, neg_k=5, seed=0):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.dim = dim
        self.lr = lr
        self.neg_k = neg_k
        self.seed = seed

        self.w_in = (rng.random((vocab_size, dim)) - 0.5) / dim
        self.w_out = np.zeros((vocab_size, dim), dtype=np.float64)

    @staticmethod
    def _sigmoid(x):
        out = []
        pos = out >= 0
        neg = ~pos
        out[pos] = 1 / (1 + np.exp(-x[pos]))
        expx = np.exp(x[neg])
        out[neg] = expx / (1 + expx)
        return out

    def train_batch(self, centers, contexts, negs):
        B = centers.shape[0]
        K = neg.shape[1]

        v_c = self.w_in[centers]             #(B,D)             
        u_o = self.w_out[contexts]           #(B,D)
        u_k = self.w_out[negs]               #(B,K,D)

        pos_score = np.sum(v_c * u_o, axis=1)            #(B,)
        pos_sig = SGNS._sigmoid(pos_score)               #(B,)
        


        neg_score = -np.einsum("bd,bkd->bk", v_c, u_k)   #(B,K)
        neg_sig = SGNS._sigmoid(neg_score)               #(B,K)

        # dL / d neg_score
        # dL / d pos_score 
        grad_pos = pos_sig - 1          #(B,)
        grad_neg = neg_sig - 1          #(B,K)

        loss = - np.log(pos_sig + 1e-12).mean() - np.sum(np.log(neg_sig + 1e-12), axis=1).mean()

        grad_v_c = np.einsum("bd,b->bd", u_o, grad_pos) + np.einsum("bkd,bk->bd", u_k, -neg_sig)
        grad_u_o = np.einsum("bd,b->bd", v_c, grad_pos)
        grad_u_k = np.einsum("bd,bk->bkd", v_c, -grad_neg)                                         #(B,K,D)

        self.w_in[centers] -= self.lr * grad_v_c
        self.w_out[contexts] -= self.lr * grad_u_o

        neg_ids = negs.reshape(-1)
        grad_u_k_flat = grad_u_k.reshape(-1, self.dim)

        for wid, g in zip(neg_ids, grad_u_k_flat):
            self.w_out[wid] -= self.lr * g

        return float(loss)
    
    def get_vector(self, wid):
        return self.w_in(wid)
    
    def most_similar(self, query_wid, topn=10):
        q = self.w_in[query_wid]
        w = self.w_in

        demon = np.linalg.norm(w, axis=1) * (np.linalg.norm(q) + 1e-12) + 1e-12
        cos = w @ q / demon

        cos[query_wid] = -1
        top = np.partition(-cos, topn)[:topn]
        top = top[np.argsort(-cos[top])]
        return list(zip(top.tolist(), cos[top].tolist()))

# Training loop 

