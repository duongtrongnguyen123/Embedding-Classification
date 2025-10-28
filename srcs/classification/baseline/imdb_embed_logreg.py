from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Sequence, Optional, Set

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import unicodedata

# -----------------------------
# Tokenizer
# -----------------------------

_tok_re = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*(?:\s?['’]\s?[A-Za-z]+)?|\d+|[.!?]")
_year_re = re.compile(r"^(1|2)\d{3}$")

_QUOTE_TRANS = str.maketrans({
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
})

def normalize_text(s: str) -> str:
    return unicodedata.normalize("NFKC", s).translate(_QUOTE_TRANS)

_CONTRACTIONS = (
    ("n't", ("not",)),
    ("'re", ("are",)),
    ("'ll", ("will",)),
    ("'ve", ("have",)),
    ("'m",  ("am",)),
    ("'d",  ("would",)),
    ("'s",  ()),  
)

def expand_contraction(tok: str) -> List[str]:
    for suf, exp in _CONTRACTIONS:
        if len(tok) > len(suf) and tok.endswith(suf):
            base = tok[:-len(suf)]
            return [base, *exp] if exp else [base]
    return [tok]

def _norm_token(tok: str) -> str:
    if tok.isdigit():
        if _year_re.fullmatch(tok):
            return "<year>"
        elif len(tok) <= 1:
            return "<digit>"
        else:
            return "<nums>"
    return tok

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    text = normalize_text(text)
    raw = _tok_re.findall(text.lower())
    out: List[str] = []
    for t in raw:
        for tt in expand_contraction(t):
            out.append(_norm_token(tt))
    return out


AUX = {"do","does","did","am","is","are","was","were","be","been","being",
       "have","has","had","will","would","shall","should","can","could",
       "may","might","must"}
INTENS = {"really","very","quite","so","too","extremely","fairly",
          "pretty","rather","somewhat","kinda","sorta","at","all"}


def read_txts(dir_path: Path) -> List[str]:
    texts: List[str] = []
    for root, _, files in os.walk(dir_path):
        for fn in files:
            if fn.lower().endswith(".txt"):
                p = Path(root) / fn
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        texts.append(f.read())
                except Exception as e:
                    sys.stderr.write(f"[warn] failed to read {p}: {e}\n")
    return texts



def load_imdb(data_root: Path) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    train_neg = read_txts(data_root / "train" / "neg")
    train_pos = read_txts(data_root / "train" / "pos")
    test_neg  = read_txts(data_root / "test"  / "neg")
    test_pos  = read_txts(data_root / "test"  / "pos")

    X_train = train_neg + train_pos
    y_train = np.array([0]*len(train_neg) + [1]*len(train_pos), dtype=np.int64)

    X_test  = test_neg + test_pos
    y_test  = np.array([0]*len(test_neg) + [1]*len(test_pos), dtype=np.int64)

    return X_train, y_train, X_test, y_test



def load_embeddings(data_root: Path):
    emb_in  = torch.load(data_root / "embed_in.pt", map_location="cpu")
    emb_out = torch.load(data_root / "embed_out.pt", map_location="cpu")
    vocab   = torch.load(data_root / "vocab.pt", map_location="cpu")

    if isinstance(vocab, dict):
        word2id = vocab.get("word2id", vocab.get("word2idx", None))
        pair_map = vocab.get("old2new_for_pair", vocab.get("pair_map", {}))
    else:
        raise ValueError("vocab.pt must be a dict containing 'word2id' and 'old2new_for_pair'")

    if word2id is None:
        raise ValueError("vocab.pt missing 'word2id'")

    word2id = {str(k): int(v) for k, v in word2id.items()}
    if isinstance(pair_map, dict):
        pair_map = {int(k): int(v) for k, v in pair_map.items()}
    else:
        pair_map = {}

    if emb_in.shape != emb_out.shape:
        raise ValueError(f"embed_in/out shapes differ: {emb_in.shape} vs {emb_out.shape}")

    w = (emb_in + emb_out) / 2.0
    return w, word2id, pair_map

def remove_first_pc(w: torch.Tensor, k: int=1) -> torch.Tensor:
    w = w - w.mean(dim=0, keepdim=True)       
    U, S, Vh = torch.linalg.svd(w, full_matrices=False)

    pcs = Vh[:k]                              # [k, D]

    proj = (w @ pcs.T) @ pcs                  # [V, D]
    return w - proj





def aux_intens_id(word2id, aux, intens):
    ids = set()
    for w in aux:
        if w in word2id: ids.add(word2id[w])
    for w in intens:
        if w in word2id: ids.add(word2id[w])
    return ids

def encode_tokens_to_ids(tokens, word2id, pair_map, skip_after_not, not_id):
    out = []
    prev_id = -1
    prev_in_negate = False

    for tok in tokens:
        cur_id = word2id.get(tok, -1)
        if cur_id < 0:
            if not prev_in_negate:
                prev_id = -1
            continue

        if prev_id != -1:
            if not_id is not None and prev_id == not_id:
                prev_in_negate = True
                if cur_id in skip_after_not:
                    continue

            key = (int(prev_id) << 32) | int(cur_id)
            merged = pair_map.get(key, None)
            if merged is not None:
                if out: out.pop()
                out.append(int(merged))
                prev_id = -1
                prev_in_negate = False
                continue
            else:
                out.append(cur_id)
        else:
            out.append(cur_id)

        prev_id = cur_id

    return out

def docs_to_matrix(texts, w, word2id, pair_map, pool="mean", l2norm=True):
    D = w.shape[1]
    skip_ids = aux_intens_id(word2id, AUX, INTENS)
    not_id = word2id.get("not", None)

    feats = np.zeros((len(texts), D), dtype=np.float32)
    for i, doc in enumerate(texts):
        toks = tokenize(doc)
        ids = encode_tokens_to_ids(toks, word2id, pair_map, skip_ids, not_id)
        if not ids:
            continue
        emb = w[ids]  # [n, D]
        if pool == "mean":
            v = emb.mean(dim=0)
        elif pool == "sum":
            v = emb.sum(dim=0)
        elif pool == "max":
            v = emb.max(dim=0).values
        else:
            raise ValueError(f"unknown pool: {pool}")
        v = v.detach().cpu().numpy().astype(np.float32)
        if l2norm:
            n = np.linalg.norm(v) + 1e-12
            v = v / n
        feats[i] = v
    return feats



def train_eval_embed(
    data_root: Path,
    remove_first_pc_flag: bool = False,
    pool: str = "mean",
    l2norm: bool = True,
    C: float = 2.0,
    max_iter: int = 2000,
):
    print("loading texts:", data_root)
    X_train_text, y_train, X_test_text, y_test = load_imdb(data_root)
    print(f"train: {len(X_train_text):,} | test: {len(X_test_text):,}")

    print("loading embeddings and vocab...")
    w, word2id, pair_map = load_embeddings(data_root)
    print(f"w shape: {tuple(w.shape)} | vocab: {len(word2id):,} | pair_map: {len(pair_map):,}")

    if remove_first_pc_flag:
        print("removing first principal component from w...")
        w = remove_first_pc(w)

    print("ncoding documents -> features...")
    Xtr = docs_to_matrix(X_train_text, w, word2id, pair_map, pool=pool, l2norm=l2norm)
    Xte = docs_to_matrix(X_test_text,  w, word2id, pair_map, pool=pool, l2norm=l2norm)
    print(f"[info] Xtr: {Xtr.shape} | Xte: {Xte.shape}")


    clf = LogisticRegression(
        solver="lbfgs",
        C=C,
        max_iter=max_iter,
        n_jobs=-1,
        verbose=0,
    )
    clf.fit(Xtr, y_train)

    print("evaluating...")
    y_pred = clf.predict(Xte)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("\n[report]\n", classification_report(y_test, y_pred, digits=4))






def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IMDB classification using precomputed embeddings (no TF-IDF)")
    p.add_argument("--data_root", type=str, default="./data",
                   help="Folder with embed_in.pt, embed_out.pt, vocab.pt and IMDB train/test (default: ./data)")
    p.add_argument("--remove_first_pc", action="store_true",
                   help="Remove the first principal component from word embeddings before pooling")
    p.add_argument("--pool", type=str, choices=["mean","sum","max"], default="mean",
                   help="Pooling type over tokens (default: mean)")
    p.add_argument("--no_l2norm", action="store_true",
                   help="Disable L2-normalization of document vectors")
    p.add_argument("--C", type=float, default=2.0, help="Inverse regularization for LogisticRegression")
    p.add_argument("--max_iter", type=int, default=2000)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--save", action="store_true")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    _ = train_eval_embed(
        data_root=Path(args.data_root),
        remove_first_pc_flag=args.remove_first_pc,
        pool=args.pool,
        l2norm=not args.no_l2norm,
        C=args.C,
        max_iter=args.max_iter,
    )

if __name__ == "__main__":
    main()
