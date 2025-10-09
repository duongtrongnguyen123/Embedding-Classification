import random, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial 

import matplotlib.pyplot as plt
import os
import time

torch.set_num_threads(8)               #nums of core
torch.set_num_interop_threads(1)

from data_pipe import (
    load_data, iter_wiki_tokens,subsample_tokens, 
    SkipGramPairsIterable, SkipGramDataset,
    NegativeSampler, make_collate_fn
)
    


class SGNS(nn.Module):
    def __init__(self, vocab_size=None, dim: int=None, padding_idx=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed_in = nn.Embedding(vocab_size, dim, padding_idx=padding_idx, sparse=True)
        self.embed_out = nn.Embedding(vocab_size, dim, padding_idx=padding_idx, sparse=True)
        nn.init.uniform_(self.embed_in.weight, -0.5/vocab_size, 0.5/vocab_size)
        nn.init.zeros_(self.embed_out.weight)

    def forward(self, centers, pos, neg):
        v_c = self.embed_in(centers)
        u_o = self.embed_out(pos)
        u_k = self.embed_out(neg)


        pos_score = (v_c * u_o).sum(dim=1)

        # if cpu:
        neg_score = (u_k * v_c.unsqueeze(1)).sum(dim=-1)
        # if gpu:
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

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_dir = os.path.join(curr_dir, "..", "data", "vocab.pt")

    vocab = torch.load(vocab_dir, map_location='cpu')
    print("loading vocab...")
    word2id = vocab["word2id"]
    id2word = vocab["id2word"]
    counts = vocab["counts"]
    count = vocab["count"]
    keep_probs = vocab["keep_probs"]
    print(len(word2id))

    keep_probs[word2id["<unk>"]] = 1         #unk:dummy vector
    print("loaded vocab...")
    def make_sub_token_iter():                      # moi lan goi make_sub phai tra 1 iter moi. Nen k duoc dung iter co dinh train_token_iter
        return subsample_tokens(iter_wiki_tokens("train"),
                                 word2id, keep_probs, "<unk>",
                                   rng=random.Random(123))
    def make_sub_token_valid_iter():
        return subsample_tokens(iter_wiki_tokens("validation"),
                                 word2id, keep_probs, "<unk>",
                                   rng=random.Random(123))


    #Build iter dataset for DataLoader
    pairs_iterable = SkipGramPairsIterable(make_sub_token_iter, window=5, rng=random.Random(1234), max_pair=None)         #Bo dau ngoac 
    valid_pairs_iterable = SkipGramPairsIterable(make_sub_token_valid_iter, window=5, rng=random.Random(111), max_pair=None)
    dataset = SkipGramDataset(pairs_iterable)
    valid_dataset = SkipGramDataset(valid_pairs_iterable)

    neg_sampler = NegativeSampler(counts.to("cpu"), device="cpu")
    collate_fn = partial(make_collate_fn, neg_sampler=neg_sampler, neg_k=15, avoid_self=True)     #collate dung chung duoc chi can thay dataset vi no chi quy dinh cach gom batch 

    loader = DataLoader(dataset, batch_size=32768, collate_fn=collate_fn,
                        num_workers=0, pin_memory=False)  #pin_mem=True neu GPU
    valid_loader = DataLoader(valid_dataset, batch_size=32768, collate_fn=collate_fn,
                              num_workers=0, pin_memory=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SGNS(len(id2word), dim=192).to(device)
    opt = torch.optim.SparseAdam(model.parameters(), lr=5e-3)

    model.train()
    lossplot = LossPlotter()
    for epoch in range(3):
        train_loss = 0.0
        total_sample = 0
        start = time.time()
        for step, batch in enumerate(loader):
            center = batch["center"].to(device)#, nonblocking=True
            pos = batch["pos"].to(device)#, nonblocking=True
            neg = batch["neg"].to(device)#, nonblocking=True
            B = center.shape[0]
            
            opt.zero_grad(set_to_none=True)
            loss = model(center, pos, neg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss += loss.item() * B
            total_sample += B

            if step % 100 == 0:
                elapsed = time.time() - start
                start = time.time()
                print(f"time: {elapsed}")
                print(f"step {step}: loss={loss.item():.4f}")
        train_loss = train_loss / total_sample

        valid_loss = 0.0
        total_sample = 0
        for batch in valid_loader:
            center = batch["center"].to(device)
            pos = batch["pos"]
            neg = batch["neg"]
            B = center.shape[0]

            loss = model(center, pos, neg)
            valid_loss += loss
            total_sample += B
        valid_loss = valid_loss / total_sample
        evaluate(model, word2id, id2word, wid=["man", "woman", "king", "queen", "happy", "good", "bad", "nice", "time"])
        lossplot.update(epoch, train_loss, valid_loss)


        bundle = {
            "word2id" : word2id,
            "id2word" : id2word,
            "w_in" : model.embed_in.weight.detach().cpu().float(),
            "w_out" : model.embed_out.weight.detaach().cpu().float()
        }    
        torch.save(bundle, f"/kaggle/working/w2v_epoch{epoch}.pt")














#8/10