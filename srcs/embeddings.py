import os
import math 
import random 
from collections import Counter, defaultdict

import numpy as np

def load_text8_tokens():
    with open("text8", "r") as f:
        tokens = f.read().split()
    return tokens

def phrase_pass(tokens,
                min_count_unigram=5,
                min_count_bigram=10,
                delta=5.0,
                threshold=100.0,
                sep="_"):
    """
    One pass of word2phrase (bigram merge).
    tokens: list[str] (đã lowercase, clean)
    Trả về: tokens_merged (list[str])
    """

    N = len(tokens)
    # 1) Count unigrams and bigrams
    uni = Counter(tokens)
    bigrams = Counter(zip(tokens[:-1], tokens[1:]))

    # 2) Build merge set by PMI-like score
    #    score = ((cnt12 - delta) / (cnt1 * cnt2)) * N
    merge_pairs = set()
    for (w1, w2), c12 in bigrams.items():
        if c12 < min_count_bigram:
            continue
        c1, c2 = uni[w1], uni[w2]
        if c1 < min_count_unigram or c2 < min_count_unigram:
            continue
        score = ((c12 - delta) / (c1 * c2)) * N
        if score > threshold:
            merge_pairs.add((w1, w2))

    # 3) Greedy merge
    merged = []
    i = 0
    L = len(tokens)
    while i < L:
        if i < L - 1 and (tokens[i], tokens[i+1]) in merge_pairs:
            merged.append(tokens[i] + sep + tokens[i+1])
            i += 2
        else:
            merged.append(tokens[i])
            i += 1

    return merged

def word2phrase(tokens,
                passes=2,
                min_count_unigram=5,
                min_count_bigram=10,
                delta=5.0,
                threshold=100.0,
                sep="_"):
    out = tokens
    for _ in range(passes):
        out = phrase_pass(out,
                          min_count_unigram=min_count_unigram,
                          min_count_bigram=min_count_bigram,
                          delta=delta,
                          threshold=threshold,
                          sep=sep)
    return out

def build_vocab(tokens, min_count=5):
    freq = Counter(tokens)
    vocab = [w for w, i in freq.items() if i>=min_count]
    vocab = sorted(vocab, key=lambda w:-freq[w])                  #ordered


    word2id = {w:i for i, w in enumerate(vocab)}                   #ordered
    id2word = [w for w in vocab]                                   #ordered
    counts = np.array([freq[q] for q in vocab], dtype=np.int64)    #ordered
    count = counts.sum()
    return word2id, id2word, counts, count

def sub_sample(tokens, word2id, counts, total_count, t=1e-5):
    freqs = counts / total_count
    prob = np.minimum(1, (np.sqrt(t / freqs) + t / freqs))
    out = []

    for w in tokens:
        if w in word2id:
            id = word2id[w]
            if prob[id] > random.random():
                out.append(w)
    
    return out

# Build training pair
def build_skipgram_pairs(tokens, word2id, window_size, max_pair):
    pairs = []
    n = len(tokens)

    for i, w in enumerate(tokens):
        if w not in word2id:
            continue
        center = word2id[w]
        win = random.randint(1, window_size)
        left = max(0, i - win)
        right = min(n, i + win + 1)
        for j in range(left, right):
            if j == i:
                continue
            if tokens[j] not in word2id:
                continue
            pairs.append((center, word2id[tokens[j]]))
        if len(pairs) > max_pair:
            break
    random.shuffle(pairs)
    return np.array(pairs, dtype=np.int64)

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

def print_pair_for(pairs, w, word2id, id2word):
    id = word2id[w]
    
    for x, y in pairs:
        if x == id:
            print(w, "->", id2word[y])
    

class SGNS:
    def __init__(self, vocab_size, dim=100, lr=0.025, neg_k=5, seed=0):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.dim = dim
        self.lr = lr
        self.neg_k = neg_k
        self.seed = seed

        self.w_in = (rng.random((vocab_size, dim)) - 0.5) / dim
        self.w_out = (rng.random((vocab_size, dim)) - 0.5) / dim 
    
    @staticmethod
    def _sigmoid(x):
        out = np.empty_like(x, dtype=float)
        pos = x >= 0
        neg = ~pos
        out[pos] = 1 / (1 + np.exp(-x[pos]))
        expx = np.exp(x[neg])
        out[neg] = expx / (1 + expx)
        return out

    def train_batch(self, centers, contexts, negs):
        b = centers.shape[0]
        k = negs.shape[1]

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

        grad_v_c = np.einsum("bd,b->bd", u_o, grad_pos) + np.einsum("bkd,bk->bd", u_k, -grad_neg)
        grad_u_o = np.einsum("bd,b->bd", v_c, grad_pos)
        grad_u_k = np.einsum("bd,bk->bkd", v_c, -grad_neg)                                         #(B,K,D)

        np.add.at(self.w_in,  centers, -self.lr * grad_v_c)
        np.add.at(self.w_out, contexts, -self.lr * grad_u_o)
        #self.w_in[centers] -= self.lr * grad_v_c
        #self.w_out[contexts] -= self.lr * grad_u_o

        neg_ids = negs.reshape(-1)
        grad_u_k_flat = grad_u_k.reshape(-1, self.dim)

        np.add.at(self.w_out, neg_ids, -self.lr * grad_u_k_flat)

        return float(loss)
    
    def get_vector(self, wid):
        return (self.w_in[wid] + self.w_out[wid]) / 2
    
    def most_similar(self, query_wid, topn=5):
        q = self.w_in[query_wid]
        w = self.w_in

        demon = np.linalg.norm(w, axis=1) * (np.linalg.norm(q) + 1e-12) + 1e-12
        cos = w @ q / demon

        cos[query_wid] = -1
        top = np.argpartition(-cos, topn)[:topn]
        top = top[np.argsort(-cos[top])]
        return list(zip(top.tolist(), cos[top].tolist()))

# Training loop 
def interate_minibatches(pairs, batch_size):
    n = pairs.shape[0]
    for i in range(0, n, batch_size):
        yield pairs[i:i+batch_size]

def train_word2vec(
    dim=100, window=5, min_count=5,
    neg_k=5, epochs=5, batch_size=1024, 
    lr=0.025, table_size=2_000_000, 
    max_pair=3_000_000, seed=0
):
    print("loading text8 token...")
    tokens = load_text8_tokens()
    print(f"Total raw tokens:{len(tokens)}")


    print("tokens phrased...")
    tokens_phrased = word2phrase(
        tokens,
        passes=2,                # 1–2 pass là vừa
        min_count_unigram=10,    # text8: bắt đầu 10
        min_count_bigram=20,     # text8: bắt đầu 20
        delta=5.0,
        threshold=100.0,
        sep="_"
    )



    print("Building vocab...")
    word2id, id2word, counts, count = build_vocab(tokens_phrased, min_count)
    print(f"Vocab size(min_count = {min_count}) : {len(word2id)}")

    print("Subsampling...")
    tokens_sub = sub_sample(tokens_phrased, word2id, counts, count)
    print(f"Subsampling size:{len(tokens_sub)}")

    print("Builing skipgram...")
    pairs = build_skipgram_pairs(tokens_sub, word2id, window, max_pair)
    print(f"Training pairs: {len(pairs)}")

    print("Making table..")
    neg_table = make_neg_table(counts, table_size)
    print(f"Table size: {len(neg_table)}")

    model = SGNS(vocab_size=len(word2id), dim=dim,
                 lr=lr, neg_k=neg_k, seed=seed)
    
    total_step = math.ceil(len(pairs) / batch_size) * epochs
    step = 0
    rng = np.random.default_rng(seed)

    for i in range(epochs):
        idx = rng.permutation(len(pairs))
        pairs = pairs[idx]

        running = 0.0
        n_batches = 0

        for mb in interate_minibatches(pairs, batch_size):
            centers = mb[:, 0]
            contexts = mb[:, 1]
            negs = sample_negatives(neg_table, len(mb) * neg_k).reshape(len(mb), neg_k)

            loss = model.train_batch(centers, contexts, negs)
            running += loss
            n_batches += 1
            step += 1

            model.lr = lr * max(0.001, 1 - step / total_step)

            if n_batches % 200 == 0:
                print(f"epoch {i} | batch {n_batches} | loss {running / n_batches:.4f}")
        print(f"Epoch {i} done . avg loss = {running / max(1, n_batches):.4f}")
    return model, word2id, id2word

# Save, demo
def save_vectors(model, id2word, path="w2v.brown.vec"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{model.vocab_size} {model.dim}\n")
        for wid in range(model.vocab_size):
            w = id2word[wid]
            vec = model.get_vector(wid)
            f.write(w + " " + " ".join(f"{x:.6f}" for x in vec) + "\n")
    print(f"saved to {path}")

def show_similar(model, word2id, id2word, queries=("man", "woman", "king", "queen", "music", "time")):
    for q in queries:
        if q not in word2id:
            print(f"{q} is not in vocab")
            continue
        wid = word2id[q]
        similar = model.most_similar(wid, 5)
        print(f"\nMost similar to {q}")
        for wid2, cos in similar:
            print(f"Word: {id2word[wid2]}, Cos: {cos}")

if __name__ == "__main__":
    model, word2id, id2word = train_word2vec(
        dim=150, window=3, min_count=5,
        neg_k=10, epochs=10, batch_size=1024,
        lr=0.025, table_size=2_000_000,
        max_pair=3_000_000, seed=42
    )
    print_pair_for()
    print(id2word[:10])  # xem 10 id đầu map ra từ gì
    #save_vectors(model, id2word, path="w2v_brown.vec")
    show_similar(model, word2id, id2word, queries=("man", "woman", "king", "queen", "music", "time"))

    




def load_brown_tokens():
    import nltk
    try: 
        from nltk.corpus import brown
        nltk.data.path.append("/Users/hduong/Documents/foundationofai/homework/spam/srcs/ntlk_data")
    except LookupError:
        nltk.download("brown")
        from nltk.corpus import brown
        nltk.data.path.append("/Users/hduong/Documents/foundationofai/homework/spam/srcs/ntlk_data")
    except Exception:
        nltk.download("brown")
        from nltk.corpus import brown
        nltk.data.path.append("/Users/hduong/Documents/foundationofai/homework/spam/srcs/ntlk_data")

    sents = brown.sents()
    tokens = []
    for sent in sents:
        for w in sent:
            w = w.lower()

            if any(ch.isalnum for ch in w):
                tokens.append(w)
    return tokens

        
