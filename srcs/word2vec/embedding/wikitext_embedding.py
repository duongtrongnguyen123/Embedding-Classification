import os
import multiprocessing as mp
import random, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial 

import matplotlib.pyplot as plt
import time



from data_pipeline.data_pipe import (
    SkipGramPairsIterable, SkipGramDataset,
    make_collate_fn, IterFactory   
)


class SGNS(nn.Module):
    def __init__(self, vocab_size=None,*, neg_k=10, dim: int=None, counts: torch.Tensor=None, padding_idx=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.neg_k = neg_k
        self.embed_in = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        self.embed_out = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        nn.init.uniform_(self.embed_in.weight, -0.5/vocab_size, 0.5/vocab_size)
        nn.init.zeros_(self.embed_out.weight)
        p = counts.float().pow(0.75)
        p = p / p.sum()
        self.register_buffer("unigram75", p)

    @torch.no_grad()
    def neg_sample(self, B, generator=None):
        idx = torch.multinomial(self.unigram75, B*self.neg_k, replacement=True,generator=generator)
        return idx.view(B, self.neg_k).to(dtype=torch.long, device=device)
    
    def forward(self, centers, pos, generator=None):
        assert centers.dtype == torch.long and pos.dtype == torch.long
        neg = self.neg_sample(centers.size(0), generator)
        v_c = self.embed_in(centers)
        u_o = self.embed_out(pos)
        u_k = self.embed_out(neg)

        pos_score = (v_c * u_o).sum(dim=1)

        neg_score = (u_k * v_c.unsqueeze(1)).sum(dim=-1)
        #neg_score = torch.bmm(u_k, v_c.unsqueeze(2)).squeeze(2)

        return -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(dim=1)).mean()
    @torch.no_grad()
    def get_input_vectors(self):
        return self.embed_in.weight.detach().clone()
    @torch.no_grad()
    def get_output_vector(self):
        return self.embed_out.weight.detach().clone()
    
    @torch.no_grad()
    def most_similar(self, wid_ids, topn=5, use='input'):
        if use == 'input':
            w = self.embed_in.weight
        if use == 'output':
            w = self.embed_out.weight
        else:
            w = (self.embed_in.weight + self.embed_out.weight) / 2

        x = w[wid_ids]
        if x.dim() == 1: x = x.unsqueeze(0)
        w_norm = w / (w.norm(dim=1, keepdim=True) + 1e-9)
        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-9)
        cos = x_norm @ w_norm.T
        for wid in wid_ids:
            cos[:, wid] = -1.0
        vals, idx = torch.topk(cos, k=topn, dim=1)
        return vals, idx


def evaluate(model: SGNS, word2id, id2word, wid=["man", "woman", "king", "queen", "happy", "good", "bad", "nice", "time"]):
    ids = [word2id[i] for i in wid]
    cos, idx = model.most_similar(ids, 5, "input")
    for w, row_idx, row_val in zip(wid, idx, cos):
        print(f"Most similar to {w}:")
        for i, v in zip(row_idx, row_val):
            word = id2word[i.item()]
            print(f"  {word:10s}  Cos: {v.item():.3f}")
        print()
    
class LossPlotter:
    def __init__(self):
        self.train_loss = []
        self.valid_loss = []
    
    def update(self, epoch, new_tloss, new_vloss):
        self.train_loss.append(new_tloss)
        self.valid_loss.append(new_vloss)

        plt.clf()
        plt.plot(range(1, len(self.train_loss)+1), self.train_loss, marker='o', label="Train")
        plt.plot(range(1, len(self.valid_loss)+1), self.valid_loss, marker='s', label="valid")
        plt.xlabel(f"{epoch}")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid = True
        plt.pause(0.01)




if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_dir = os.path.join(curr_dir, "..", "..", "data", "vocab.pt")
    vocab = torch.load(vocab_dir, map_location='cpu')
    print("loading vocab...")
    word2id = vocab["word2id"]
    id2word = vocab["id2word"]
    counts = vocab["counts"]
    count = vocab["count"]
    keep_probs = vocab["keep_probs"]

    print("loaded vocab...")

    #numworker > 0 k goi ham long
    train_iter = IterFactory("train")
    valid_iter = IterFactory("validation")
    pairs_iterable = SkipGramPairsIterable(train_iter, window=5, rng=random.Random(1234), keep_probs=keep_probs, word2id=word2id)         #Bo dau ngoac 
    valid_pairs_iterable = SkipGramPairsIterable(valid_iter, window=5, rng=random.Random(111), keep_probs=keep_probs, word2id=word2id)
    dataset = SkipGramDataset(pairs_iterable)
    valid_dataset = SkipGramDataset(valid_pairs_iterable)

    collate_fn = partial(make_collate_fn)     #collate dung chung duoc chi can thay dataset vi no chi quy dinh cach gom batch 

    #numworker lon phai tang shards
    #numworker bottlenack
    loader = DataLoader(dataset, batch_size=32768, collate_fn=collate_fn,
                        num_workers=2, pin_memory=False, persistent_workers=True, 
                        prefetch_factor=4)  #pin_mem=True neu GPU      persistent True, prefector > 0 neu worker>0             
    valid_loader = DataLoader(valid_dataset, batch_size=32768, collate_fn=collate_fn,
                              num_workers=1, pin_memory=False, persistent_workers=True,
                                prefetch_factor=4)  #pin_mem=True neu GPU
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SGNS(len(word2id), neg_k=10, counts= counts, dim=256).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    lossplot = LossPlotter()

    start = time.time()
    base_seed=111
    for epoch in range(10):
        model.train()
        generator = torch.Generator(device=device).manual_seed(epoch+base_seed)

        train_loss = 0.0
        total_sample = 0
        for step, batch in enumerate(loader):
            center = batch["center"]
            pos = batch["pos"]
            B = center.shape[0]
            
            opt.zero_grad(set_to_none=True)
            loss = model(center, pos, generator)      #loss:tensor
            loss.backward()
            opt.step()
            train_loss += loss.item() * B
            total_sample += B

            if step % 100 == 0:
                end = time.time()
                elapsed = end - start
                start = end
                print(f"time: {elapsed}")
                print(f"step {step}: loss={loss.item():.4f}")
        print(f"total_sample : {total_sample}")
        train_loss = train_loss / total_sample

        model.eval()
        valid_loss = 0.0
        total_sample = 0
        for batch in valid_loader:
            center = batch["center"].to(device)
            pos = batch["pos"]
            B = center.shape[0]

            loss = model(center, pos, generator)
            valid_loss += loss.item() * B
            total_sample += B   
        valid_loss = valid_loss / total_sample
        evaluate(model, word2id, id2word, wid=["man", "woman", "king", "queen", "happy", "good", "bad", "nice", "time"])
        lossplot.update(epoch, train_loss, valid_loss)


        bundle = {
            "word2id" : word2id,
            "id2word" : id2word,
            "w_in" : model.embed_in.weight.detach().cpu().float(),
            "w_out" : model.embed_out.weight.detach().cpu().float()
        }    
        save_emb_path = os.path.join(curr_dir, f"embed_epoch{epoch}")
        torch.save(bundle, save_emb_path)
    












#8/10