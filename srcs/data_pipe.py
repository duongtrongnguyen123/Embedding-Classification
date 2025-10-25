from datasets import load_dataset
import random, torch
from dataclasses import dataclass

from collections import deque
from typing import Iterable, Dict, Tuple, List
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

import re
import unicodedata

_tok_re = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*(?:\s?['’]\s?[A-Za-z]+)?|\d+|[.!?]")
_year_re = re.compile(r"^(1|2)\d{3}$")

ROMAN_SMALL = {"i","ii","iii","iv","v","vi","vii","viii","ix","x","xi","xii","xiii","xiv","xv","xvi","xvii","xviii","xix","xx"}

AUX = {"do","does","did","am","is","are","was","were","be","been","being",
       "have","has","had","will","would","shall","should","can","could",
       "may","might","must"}
INTENS = {"really","very","quite","so","too","extremely","fairly","pretty",
          "rather","somewhat","kinda","sorta","at","all"}


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return (s.replace("’", "'").replace("‘", "'")
             .replace("“", '"').replace("”", '"'))



def tokenize(s: str):
    if not s:
        return []
    s = normalize_text(s)
    return _tok_re.findall(s.lower())


def expand_contraction(tok: str):
    t = tok.replace(" ", "")
    if t.endswith("n't") and len(t) > 3: return [t[:-3], "not"]
    if t.endswith("'re") and len(t) > 3: return [t[:-3], "are"]
    if t.endswith("'ll") and len(t) > 3: return [t[:-3], "will"]
    if t.endswith("'ve") and len(t) > 3: return [t[:-3], "have"]
    if t.endswith("'m")  and len(t) > 2: return [t[:-2], "am"]
    if t.endswith("'d")  and len(t) > 2: return [t[:-2], "would"] 
    if t.endswith("'s")  and len(t) > 2: return [t[:-2]]         
    return [t]

END = {".", "!", "?"}

def _norm_token(tok: str) -> str:
    if tok.isdigit():
        if _year_re.fullmatch(tok):
            return "<year>"
        return "<digit>" if len(tok) <= 1 else "<nums>"
    if tok in ROMAN_SMALL:
        return "<century>"
    return tok

def iter_wiki_sentences(split, streaming=False):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, streaming=streaming)
    sent = []

    def flush():
        nonlocal sent
        if sent:
            yield list(sent)
            sent.clear()

    for ex in ds:
        t = ex["text"]
        if not t or not t.strip():
            for s in flush():
                yield s
            continue

        for tok in tokenize(t):
            if tok in END:
                for s in flush():
                    yield s
                continue
            
        
            for st in expand_contraction(tok):
                st = _norm_token(st)
                sent.append(st)

    for s in flush():
        yield s








""" 123
def iter_wiki_tokens(split, streaming=False):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1",
                    split=split, streaming=streaming)
    
    for ex in ds:
        t = ex["text"]
        if t is None or not t.strip():
            continue
        seen = False
        for tok in tokenize(t):
            if tok is None or not tok.strip():
                continue
            if tok in {".", "!", "?"}:
                seen = True
                yield EOS
                continue

            if tok.isdigit():
                if _year_re.fullmatch(tok):
                    yield "<year>"
                    continue
                elif len(tok) <= 1:
                    yield "<digit>"
                    continue
                yield "<nums>"
                continue
            yield tok

"""

#nhận iter gốc, chỉ để tạo sx lấy pair cho skipgram 
"""1
def subsample_tokens(token_iter: Iterable[str],
                     word2id: Dict[str,int],
                     keep_probs: torch.Tensor,
                     unk_token: str = "<unk>",
                     eos_id = None,
                     rng: random.Random = None):
    if rng is None:
        rng = random.Random()
    unk_id = word2id.get(unk_token, 1)
    kp = keep_probs.detach().cpu().numpy()

    for w in token_iter:
        if w is None:
            continue
        wid = word2id.get(w)
        if wid is None or wid == unk_id:
            continue
        if wid == eos_id:
            yield eos_id
            continue
        if rng.random() <= float(kp[wid]):
            yield wid
"""

@dataclass
class IterFactory:
    split: str
    def __call__(self): 
        return iter_wiki_sentences(self.split)
            


class SkipGramPairsIterable:            
    def __init__(self, ids_iter_factory, window=5, rng=None, keep_probs: torch.Tensor=None, word2id:Dict=None):
        self.ids_iter_factory = ids_iter_factory       #khi goi yield tung id (train_ids)
        self.window = window
        self.rng = rng if rng else random.Random()
        self.keep_probs = keep_probs
        self.base_seed = 121
        self.word2id = word2id

    def __iter__(self):
        info = get_worker_info()
        if info is None:
            wid, nw = 0, 1
        else:
            wid, nw = info.id, info.num_workers

        rng = random.Random(self.base_seed ^ (0x9E3779B97F4A7C15 & (wid + 1)))

        def take_this_worker(stream):
            for i, sent in enumerate(stream):
                if (i % nw) == wid:
                    yield sent
        
        def _keep(i):
            p = self.keep_probs[i]
            return rng.random() <= p

        for sents in take_this_worker(self.ids_iter_factory()):
            ids = [self.word2id[w] for w in sents if (w in self.word2id) and _keep(self.word2id[w])]

            for center_idx in range(0, len(ids)):
                win = self.rng.randint(1, self.window)
                left = max(0, center_idx - win)
                right = min(len(ids), center_idx + win + 1)
                for j in range(left, right):
                    if j == center_idx:
                        continue
                    yield (ids[center_idx], ids[j])
            
                    


""" 123
@dataclass
class SubIterFactory:
    split: str
    word2id: Dict[str,int]
    keep_probs: torch.Tensor
    unk: str = "<unk>"
    eos_id: int = 0
    seed: int = 123
    def __call__(self):
        rng = random.Random(self.seed)
        return subsample_tokens(iter_wiki_tokens(self.split, streaming=True),
                                self.word2id, self.keep_probs, self.unk, self.eos_id, rng)


#Lay pair tu subsample
class SkipGramPairsIterable:            
    def __init__(self, ids_iter_factory, window=5, rng=None, eos_id=None, keep_probs: torch.Tensor=None):
        self.ids_iter_factory = ids_iter_factory       #khi goi yield tung id (train_ids)
        self.window = window
        self.rng = rng if rng else random.Random()
        self.eos_id = eos_id
        self.keep_probs = keep_probs

    def __iter__(self):
        ids_iter = self.ids_iter_factory()
        info = get_worker_info()
        if info is None:
            def take_this_worker(stream):
                for i, x in enumerate(stream):
                    yield x
            rng = self.rng
        else:
            wid, nw = info.id, info.num_workers
            def take_this_worker(stream):
                for i, x in enumerate(stream):
                    if i % nw == wid:
                        yield x
            if self.rng is None:
                rng = random.Random(0xABCD12 ^ wid)
            else:
                rng = random.Random(self.rng.randint(0, 2**32 - 1) ^ wid)   #rng từng worker phải khác nhau vì worker lấy tuần tự theo modulo
        src = take_this_worker(ids_iter)

        buf = deque(maxlen=2*self.window+1)
        def fill():
            try:
                while len(buf) < 2*self.window:
                    w = next(src)
                    if w == self.eos_id:
                        buf.clear()
                        continue
                    buf.append(w)
                return True
            except StopIteration:
                return False
            
        if not fill():
            return
        
        while True:
            try:
                w = next(src)
                while w == self.eos_id:
                    buf.clear()
                    if not fill():
                        return
                    w = next(src)
                buf.append(w)
            except StopIteration:
                return
            
            center_idx = self.window
            c = buf[center_idx]
            if self.keep_probs[c] < rng.random():
                continue
            win = rng.randint(1, self.window)

            for j in range(center_idx-win, center_idx+win+1):
                if j==center_idx:
                    continue
                if buf[j] == self.eos_id:
                    for _ in range(j+1):
                        buf.popleft()
                    if not fill():
                        return
                    break
                if self.keep_probs[buf[j]] > rng.random():
                    yield (buf[center_idx], buf[j])
"""


class SkipGramDataset(IterableDataset):                        #wrap SkipGramIterable de tao nhieu factory cho nhieu epoch
    def __init__(self, pairs_iterable: SkipGramPairsIterable):
        super().__init__()
        self.pairs_iterable = pairs_iterable
    
    def __iter__(self):
        yield from iter(self.pairs_iterable)    # tao iterator cho cac factory khac nhau 


def make_collate_fn(batch:List[Tuple[int,int]]):
    centers = torch.tensor([c for c,_ in batch], dtype=torch.long, device='cpu')
    pos     = torch.tensor([p for _,p in batch], dtype=torch.long, device='cpu')

    return {"center" : centers, "pos": pos}












#7/10
