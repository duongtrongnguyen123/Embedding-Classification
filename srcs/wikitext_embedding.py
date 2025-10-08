from datasets import load_dataset
from collections import Counter, deque
import math, random
from typing import Iterable, Dict, Tuple, List, Callable

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

def tokenize(s: str):
    return s.strip().split()

def iter_wiki_tokens(split, streaming=False):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1",
                      split=split, streaming=streaming)
    
    for ex in ds:
        t = ex["text"]
        if t is None or not t.strip():
            continue
        for tok in tokenize(t):
            yield tok


def load_data(streaming=False, materialize=False):
    train_iter = iter_wiki_tokens("train", streaming)
    valid_iter = iter_wiki_tokens("validation", streaming)
    test_iter = iter_wiki_tokens("test", streaming)

    if materialize:
        train_iter = list(train_iter)
        valid_iter = list(valid_iter)
        test_iter = list(test_iter)

    return train_iter, valid_iter, test_iter


def build_vocab(token_iter: Iterable[str],
                min_count=5, max_vocab_size=None,
                specials: List[str]=None):
    if specials is None:
        specials = ["<unk>"]

    counter = Counter()
    total = 0
    for tok in token_iter:
        total += 1
        counter[tok] += 1

    vocab = [w for w, i in counter.items() if i >= min_count]
    vocab = sorted(vocab, key=lambda w:-counter[w])

    if max_vocab_size is not None:
        cap = max(0, max_vocab_size - len(specials))
        vocab = vocab[:cap]

    id2word = list(specials) + [w for w in vocab]
    word2id = {w:i for i, w in enumerate(id2word)}

    counts = torch.tensor([counter[w] for w in id2word], dtype=torch.long)

    return word2id, id2word, counts, total


def compute_keep_probs(counts: torch.Tensor=None,
                      total_tokens: int=None, t=1e-5) -> torch.Tensor:
    f = counts.to(torch.float64) / max(1, total_tokens)
    p = torch.empty_like(f, dtype=torch.float32)

    nz = f > 0
    p[~nz] = 0
    
    p[nz] = (torch.sqrt(f[nz]/t) + 1) * (t / f[nz])
    return torch.clamp(p, 0, 1).to(torch.float32)

def subsample_tokens(token_iter: Iterable[str],
                     word2id: Dict[str,int],
                     keep_probs: torch.Tensor,
                     unk_token: str = "<unk>",
                     rng: random.Ramdom = None):
    if rng is None:
        rng = random.Random()
    unk_id = word2id.get(unk_token, 0)

    kp = keep_probs.detach().cpu().numpy()

    for w in token_iter:
        wid = word2id.get(w, unk_id)
        if rng.random() <= float(kp[wid]):
            yield wid


class SkipGramPairsIterable:
    def __init__(self, ids_iter_factory, window=5, rng=None, max_pair=None):
        self.ids_iter_factory = ids_iter_factory       #khi goi yield tung id (train_ids)
        self.window = window
        self.rng = rng or random.Ramdom()
        self.max_pair = max_pair

    def __iter__(self):
        ids_iter = self.ids_iter_factory()
        buf = deque(maxlen=2*self.window+1)
        try:
            for _ in range(2*self.window+1):
                buf.append(next(ids_iter))
        except StopIteration:
            return 

        made=0
        while True:
            center_idx = self.window
            win = self.rng.randint(1, self.window)
            
            for j in range(center_idx-win, center_idx+win+1):
                if j==center_idx:
                    continue
                made+=1
                yield (buf[center_idx], buf[j])
            if self.max_pair and made>self.max_pair:
                break

            try:
                buf.append(next(ids_iter))
            except StopIteration:
                return 
    
class NegativeSampler:
    def __init__(self, counts, device=None):
        if device is None:
            device = counts.device
        freq = counts.to(torch.float64)
        freq = torch.clamp(freq, 0)
        p = freq.pow(0.75)
        s = p.sum()
        if s <= 0:
            p = torch.ones_like(p)
            s = p.sum()
        self.probs = (p / s).to(torch.float32).to(device)
        self.device = device
        self.V = counts.numel()

    @torch.no_grad()
    def sample(self, num_samples) -> torch.Tensor:
        return torch.multinomial(self.probs, num_samples, replacement=True)
    
class SkipGramDataset(IterableDataset):                        #wrap SkipGramIterable de tao nhieu factory cho nhieu epoch
    def __init__(self, pairs_iterable: SkipGramPairsIterable):
        super().__init__()
        self.pairs_iterable = pairs_iterable
    
    def __iter__(self):
        yield from iter(self.pairs_iterable)    # tao iterator cho cac factory khac nhau 


def make_collate_fn(neg_sampler: NegativeSampler, neg_k, avoid_self=True):
    def collate(batch: List[Tuple[int, int]]):
        centers = torch.tensor([c for c,_ in batch], dtype=torch.long)
        pos = torch.tensor([p for _,p in batch], dtype=torch.long)

        B = centers.size(0)
        neg = neg_sampler.sample(neg_k * B).view(B, neg_k)

        #neg dinh voi centers hoac pos
        if avoid_self:
            with torch.no_grad():
                flatten = neg.view(-1)
                resample = (flatten == centers.repeat_interleave(neg_k)) | (flatten == pos.repeat_interleave(neg_k))
                n_need = resample.sum()
                while n_need > 0:
                    flatten[resample] = neg_sampler.sample(n_need)
                    resample = (flatten == centers.repeat_interleave(neg_k)) | (flatten == pos.repeat_interleave(neg_k))
                    n_need = resample.sum()
                neg = flatten.view(B, neg_k)

        return {"center" : centers, "pos": pos, "neg": neg}

    return collate




class SGNS(nn.Module):
    def __init__(self, vocab_size=None, dim: int=None, padding_idx=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed_in = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        self.embed_out = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        self.reset_parameter()

    def reset_parameter(self):
        bound = 0.5 / self.dim
        nn.init.uniform__(self.embed_in.weight, -bound, bound)
        nn.init.uniform__(self.embed_out.weight, -bound, bound)

    def forward(self, centers, pos, neg):
        v_c = self.embed_in(centers)
        u_o = self.embed_out(pos)
        u_k = self.embed_out(neg)


        pos_score = (v_c * u_o).sum(dim=1)
        pos_loss = F.logsigmoid(pos_score)

        neg_score = -torch.einsum("bd,bkd->bk", v_c, u_k)
        neg_loss = F.logsigmoid(neg_score).sum(dim=1)

        loss = (-pos_loss -neg_loss).mean()

        return loss
    @torch.no_grad()
    def get_input_vectors(self):
        return self.embed_in.weight.detach().clone()
    @torch.no_grad()
    def get_output_vector(self):
        return self.embed_out.weight.detach().clone()
    
    @torch.no_grad()
    def most_similar(self, wid, topn=5, use='input'):
        if use == 'input':
            w = self.embed_in.weight
        if use == 'output':
            w = self.embed_out.weight
        else:
            w = (self.embed_in.weight + self.embed_out.weight) / 2

        x = w[wid]
        if x.dim() == 1: x = x.unsqueeze(0)
        w_norm = w / (w.norm(dim=1, keepdim=True) + 1e-9)
        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-9)
        cos = x_norm @ w_norm.T
        vals, idx = torch.topk(cos, k=topn, dim=1)
        return vals, idx

if __name__ == "__main__":

    print("load stream train...")
    #single use token iter for vocab
    train_tokens_iter = iter_wiki_tokens("train")
    print("building vocab...")
    word2id, id2word, counts, count = build_vocab(train_tokens_iter, min_count=5, max_vocab_size=None)
    print("compute keep probs...")
    keep_probs = compute_keep_probs(counts, count, t=1e-5)
    keep_probs[word2id["<unk>"]] = 1         #unk:dummy vector

    def make_sub_token_iter():                      # moi lan goi make_sub phai tra 1 iter moi. Nen k duoc dung iter co dinh train_token_iter
        return subsample_tokens(iter_wiki_tokens("train"),
                                 word2id, keep_probs, "<unk>",
                                   rng=random.Random(123))


    #Build iter dataset for DataLoader
    pairs_iterable = SkipGramPairsIterable(make_sub_token_iter, 5, rng=random.Random(1234), max_pair=None)         #Bo dau ngoac 
    dataset = SkipGramDataset(pairs_iterable)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neg_sampler = NegativeSampler(counts.to(device), device=device)
    collate_fn = make_collate_fn(neg_sampler, 10, avoid_self=True)

    loader = DataLoader(dataset, batch_size=1024, collate_fn=collate_fn,
                        num_workers=4, pin_memory=False)  #pin_mem=True neu GPU

    
    model = SGNS(len(id2word), dim=250).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    model.train()
    for step, batch in enumerate(loader):
        center = batch["center"].to(device)
        pos = batch["pos"].to(device)
        neg = batch["neg"].to(device)
        
        loss = model(center, pos, neg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f"step {step}: loss={loss.item():.4f}")









