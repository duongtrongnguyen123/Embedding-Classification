from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib



_tok_re = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*(?:\s?['’]\s?[A-Za-z]+)?|\d+|[.!?]")
_year_re = re.compile(r"^(1|2)\d{3}$")

END = {".", "!", "?"}

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

def custom_tokenize(s: str) -> List[str]:
    if not s:
        return []
    s = normalize_text(s)
    raw = _tok_re.findall(s.lower())
    out: List[str] = []
    for tok in raw:
        expanded = []
        for t in expand_contraction(tok):
            expanded.append(_norm_token(t))
        out.extend(expanded)
    return out



def _read_txts(dir_path: Path) -> List[str]:
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
    train_neg = _read_txts(data_root / "train" / "neg")
    train_pos = _read_txts(data_root / "train" / "pos")
    test_neg  = _read_txts(data_root / "test"  / "neg")
    test_pos  = _read_txts(data_root / "test"  / "pos")

    X_train = train_neg + train_pos
    y_train = np.array([0]*len(train_neg) + [1]*len(train_pos), dtype=np.int64)

    X_test  = test_neg + test_pos
    y_test  = np.array([0]*len(test_neg) + [1]*len(test_pos), dtype=np.int64)

    return X_train, y_train, X_test, y_test



def train_and_eval(
    data_root: Path,
    min_df: int | float = 2,
    max_df: float = 0.9,
    C: float = 2.0,
    max_iter: int = 1000,
) -> dict:
    print("loading IMDB data from:", data_root)
    X_train, y_train, X_test, y_test = load_imdb(data_root)

    print(f"train: {len(X_train):,} docs | test: {len(X_test):,} docs")

    print("building TF-IDF with custom tokenizer...")
    vectorizer = TfidfVectorizer(
        tokenizer=custom_tokenize,
        preprocessor=None,
        token_pattern=None,
        min_df=min_df,
        max_df=max_df,
        dtype=np.float32,
        sublinear_tf=True,
        lowercase=False,
    )
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)
    print(f"Xtr shape: {Xtr.shape}, Xte shape: {Xte.shape}")

    print("training LogisticRegression...")
    clf = LogisticRegression(
        solver="liblinear",
        C=C,
        max_iter=max_iter,
        n_jobs=None,
        verbose=0,
    )
    clf.fit(Xtr, y_train)

    print("evaluating...")
    y_pred = clf.predict(Xte)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")

    print("\n[report]\n", classification_report(y_test, y_pred, digits=4))

    return {"acc": acc, "clf": clf, "vectorizer": vectorizer}




def _parse_min_df(x: str):
    try:
        return int(x)
    except ValueError:
        return float(x)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IMDB TF-IDF + Logistic Regression using custom tokenizer (regex + contractions + number buckets)"
    )
    p.add_argument("--data_root", type=str, default="./data",
                   help="Path to ./data containing train/ and test/ with pos/ neg/ subfolders (default: ./data)")
    p.add_argument("--ngram", type=int, nargs=2, default=(1,2),
                   help="ngram range, e.g., --ngram 1 2")
    p.add_argument("--min_df", type=_parse_min_df, default=2,
                   help="min_df for TfidfVectorizer: int (>=1) or float proportion (0..1). Default 2")
    p.add_argument("--max_df", type=float, default=0.9,
                   help="max_df for TfidfVectorizer. Default 0.9")
    p.add_argument("--C", type=float, default=2.0,
                   help="Inverse regularization strength for LogisticRegression. Default 2.0")
    p.add_argument("--max_iter", type=int, default=1000,
                   help="Max iterations for LogisticRegression. Default 1000")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Where to save artifacts (vectorizer, model, metrics.json)")
    p.add_argument("--save", action="store_true",
                   help="If set, save vectorizer/model/metrics to --out_dir")

    args = p.parse_args()
    if args.out_dir is not None:
        args.out_dir = Path(args.out_dir)
    return args

def main() -> None:
    args = parse_args()
    artifacts = train_and_eval(
        data_root=Path(args.data_root),
        min_df=args.min_df,
        max_df=args.max_df,
        C=args.C,
        max_iter=args.max_iter,
    )

if __name__ == "__main__":
    main()
