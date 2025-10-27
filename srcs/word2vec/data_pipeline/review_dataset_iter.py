from datasets import load_dataset
from itertools import islice


import re
import unicodedata

_tok_re = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*(?:\s?['’]\s?[A-Za-z]+)?|\d+|[.!?]")
_year_re = re.compile(r"^(1|2)\d{3}$")

END = {".", "!", "?"}

_QUOTE_TRANS = str.maketrans({
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
})

DATA_FILES = "hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/Movies_and_TV.jsonl"


def normalize_text(s: str):
    return unicodedata.normalize("NFKC", s).translate(_QUOTE_TRANS)
    
def tokenize(s: str):
    if not s:
        return []
    s = normalize_text(s)
    return _tok_re.findall(s.lower())

_CONTRACTIONS = (
    ("n't", ("not",)),
    ("'re", ("are",)),
    ("'ll", ("will",)),
    ("'ve", ("have",)),
    ("'m",  ("am",)),
    ("'d",  ("would",)),
    ("'s",  ()),
)

def expand_contraction(tok: str):
    for suf, exp in _CONTRACTIONS:
        if len(tok) > len(suf) and tok.endswith(suf):
            base = tok[:-len(suf)]
            return [base, *exp] if exp else [base]
    return [tok]



def _norm_token(tok: str):
    if tok.isdigit():
        if _year_re.fullmatch(tok):
            return "<year>"
        elif len(tok) <= 1:
            return "<digit>"
        else:
            return "<nums>"
    return tok

def train_iter_review_sentences(streaming=True):
    ds = load_dataset("json", data_files=DATA_FILES, split="train", streaming=True)

    subset = islice(ds, 10_000, 1_600_000)

    sent = []

    def _flush():
        nonlocal sent
        if sent:
            out = sent
            sent = []
            return out
        return None

    for ex in subset:
        t = ex.get("content") or ex.get("reviewText") or ex.get("review_text") or ex.get("text") or ""
        if not t or not t.strip():
            out = _flush()
            if out:
                yield out
            continue

        for tok in tokenize(t):
            if not tok or not tok.strip():
                continue
            if tok in END:
                out = _flush()
                if out:
                    yield out
                continue
            for st in expand_contraction(tok):
                st = _norm_token(st)
                if st:
                    sent.append(st)

    out = _flush()
    if out:
        yield out



def valid_iter_review_sentences(streaming=True):
    ds = load_dataset("json", data_files=DATA_FILES, split="train", streaming=True)

    subset = islice(ds, 0, 10_000)

    sent = []

    def _flush():
        nonlocal sent
        if sent:
            out = sent
            sent = []
            return out
        return None

    for ex in subset:
        t = ex.get("content") or ex.get("reviewText") or ex.get("review_text") or ex.get("text") or ""
        if not t or not t.strip():
            out = _flush()
            if out:
                yield out
            continue

        for tok in tokenize(t):
            if not tok or not tok.strip():
                continue
            if tok in END:
                out = _flush()
                if out:
                    yield out
                continue
            for st in expand_contraction(tok):
                st = _norm_token(st)
                if st:
                    sent.append(st)

    out = _flush()
    if out:
        yield out


