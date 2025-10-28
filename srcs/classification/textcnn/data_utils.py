import os
import re
import unicodedata
from pathlib import Path

_tok_re = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*(?:\s?['’]\s?[A-Za-z]+)?|\d+|[.!?]")

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return (s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"'))

def tokenize(s: str):
    if not s: return []
    s = normalize_text(s)
    return _tok_re.findall(s.lower())

def expand_contraction(tok: str):
    t = tok.replace(" ", "").replace("’", "'")
    if t.endswith("n't"): return [t[:-3], "not"]
    if t.endswith("'re"): return [t[:-3], "are"]
    if t.endswith("'ll"): return [t[:-3], "will"]
    if t.endswith("'ve"): return [t[:-3], "have"]
    if t.endswith("'m"):  return [t[:-2], "am"]
    if t.endswith("'d"):  return [t[:-2], "would"]
    if t.endswith("'s"):  return [t[:-2]]
    return [t]

def preprocess_imdb(text):
    tokens = []
    for tok in tokenize(text):
        tokens.extend(expand_contraction(tok))
    return tokens

def load_data(folder, label):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    texts, labels = [], []
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix == ".txt":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    texts.append(f.read())
                    labels.append(label)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return texts, labels

def encode_text(text, word2id, max_len=300):
    tokens = preprocess_imdb(text)
    ids = [word2id.get(tok, word2id.get("<unk>", 0)) for tok in tokens]
    ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
    return ids