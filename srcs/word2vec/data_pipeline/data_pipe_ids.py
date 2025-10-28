import os, random
import numpy as np
import torch 
from torch.utils.data import IterableDataset, DataLoader

class SkipGramPairIterable(IterableDataset):
    def __init__(
        self,
        path,
        save_starts_path: str=None,
        train = True,
        keep_probs: torch.Tensor=None,
        window: int=5,
        batch_size: int=32768,
        seed: int=121,
        device: torch.device | None=None
    ):
        self.path = path
        self.train = train
        self.keep_probs = keep_probs.detach().to("cpu", dtype=torch.float32)
        self.window = window
        self.batch_size = batch_size
        self.seed = seed
        self.device = device

        self.encode, self.sizes = self._open_memmaps()
        self.save_starts_path = save_starts_path if save_starts_path is not None else self._compute_starts(self.sizes)


    def _open_memmaps(self):
        n_encode = np.memmap(self.path, dtype=np.uint32, mode='r', shape=(1,), offset=0)[0]
        off_ids = 4
        encoded = np.memmap(self.path, dtype=np.uint32, mode='r', shape=(n_encode,), offset=off_ids)

        off_n_size = 4 + 4*n_encode
        n_size   = np.memmap(self.path, dtype=np.uint32, mode='r', shape=(1,), offset=off_n_size)[0]
        off_sizes = 4 + off_n_size
        sizes   = np.memmap(self.path, dtype=np.uint32, mode='r', shape=(n_size,), offset=off_sizes)
        
        return encoded, sizes


    def _compute_starts(self, sizes: np.ndarray):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if self.train:
            self.save_starts_path = os.path.join(base_dir, "..", "..", "..", "data", "train_starts.bin")
        else:
            self.save_starts_path = os.path.join(base_dir, "..", "..", "..", "data", "valid_starts.bin")
        starts = np.zeros_like(sizes, dtype=np.uint32)
        np.cumsum(sizes[:-1], dtype=np.uint32, out=starts[1:])
        with open(self.save_starts_path, "wb") as f:
            starts.tofile(f)

        return self.save_starts_path


    def __iter__(self):
        encode, sizes = self._open_memmaps()
        starts = np.memmap(self.save_starts_path, dtype=np.uint32, mode='r')
        
        worker = torch.utils.data.get_worker_info()
        n_sent = sizes.shape[0]

        if worker is None:
            lo, hi, worker_id = 0, n_sent, 0
        else:
            per = (n_sent + worker.num_workers - 1) // worker.num_workers
            lo = worker.id * per
            hi = min(n_sent, lo + per)
            worker_id = worker.id

            rng = np.random.default_rng(self.seed + 9973 * worker_id)

        kp = self.keep_probs.numpy()

        batch_center = []
        batch_context = []
        for st in range(lo, hi):
            total = 0
            l = starts[st]
            r = l + sizes[st]

            sent = encode[l:r]
            mask = kp[sent] > rng.random(r-l)
            cand = np.nonzero(mask)[0]
            if mask.sum() == 0:
                continue
            keep_id = sent[cand]

            n = keep_id.shape[0]    
            for i in range(n):
                win = rng.integers(1, self.window+1)
                
                start = max(0, i - win)
                end = min(n, i + win + 1)

                for j in range(start, end):
                    if j == i:
                        continue
                    batch_center.append(keep_id[i])
                    batch_context.append(keep_id[j])
            
            if len(batch_center) >= self.batch_size:
                centers  = torch.tensor(batch_center[:self.batch_size], dtype=torch.long)
                contexts = torch.tensor(batch_context[:self.batch_size], dtype=torch.long)
                yield centers, contexts
                del batch_center[:self.batch_size]
                del batch_context[:self.batch_size]
            
            
        if len(batch_center) > 0:
            centers = torch.tensor(batch_center[:self.batch_size], dtype=torch.long)
            contexts = torch.tensor(batch_context[:self.batch_size], dtype=torch.long)
            yield centers, contexts
            del batch_center[:self.batch_size]
            del batch_context[:self.batch_size]

