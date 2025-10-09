from datasets import load_dataset
import random, torch

from collections import Counter, deque
from typing import Iterable, Dict, Tuple, List
from torch.utils.data import IterableDataset

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


    
def subsample_tokens(token_iter: Iterable[str],
                     word2id: Dict[str,int],
                     keep_probs: torch.Tensor,
                     unk_token: str = "<unk>",
                     rng: random.Random = None):
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
        self.rng = rng or random.ndom()
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
    def __init__(self, counts, device='cpu'):
        self.device = device
        freq = counts.to(device = device, dtype=torch.float64)
        freq = torch.clamp(freq, 0)
        p = freq.pow(0.75)
        s = p.sum().item()
        if s <= 0:
            p = torch.ones_like(p)
            s = p.sum()
        self.probs = (p / s).to(torch.float32).to(device)
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


def make_collate_fn(batch:List[Tuple[int,int]], *, neg_sampler: NegativeSampler, neg_k=None, avoid_self=True):
    centers = torch.tensor([c for c,_ in batch], dtype=torch.long, device='cpu')
    pos     = torch.tensor([p for _,p in batch], dtype=torch.long, device='cpu')

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












#7/10