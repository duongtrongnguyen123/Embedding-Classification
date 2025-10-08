import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial 

import matplotlib.pyplot as plt

from data_pipe import (
    load_data, iter_wiki_tokens, build_vocab, compute_keep_probs,
    subsample_tokens, SkipGramPairsIterable, SkipGramDataset,
    NegativeSampler, make_collate_fn
)

    



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
        nn.init.uniform_(self.embed_in.weight, -bound, bound)
        nn.init.uniform_(self.embed_out.weight, -bound, bound)

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
    

def evaluate(model: SGNS, id2word, wid=["man", "woman", "king", "queen", "happy", "good", "bad", "nice", "time"]):
    cos, idx = model.most_similar(wid, 5, "input")
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
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid = True
        plt.pause(0.01)

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
    def make_sub_token_valid_iter():
        return subsample_tokens(iter_wiki_tokens("validation"),
                                 word2id, keep_probs, "<unk>",
                                   rng=random.Random(123))


    #Build iter dataset for DataLoader
    pairs_iterable = SkipGramPairsIterable(make_sub_token_iter, 5, rng=random.Random(1234), max_pair=None)         #Bo dau ngoac 
    valid_pairs_iterable = SkipGramPairsIterable(make_sub_token_valid_iter, 5, rng=random.Random(111), max_pair=None)
    dataset = SkipGramDataset(pairs_iterable)
    valid_dataset = SkipGramDataset(valid_pairs_iterable)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neg_sampler = NegativeSampler(counts.to(device), device=device)
    collate_fn = partial(make_collate_fn, neg_sampler=neg_sampler, neg_k=10, avoid_self=True)     #collate dung chung duoc chi can thay dataset vi no chi quy dinh cach gom batch 

    loader = DataLoader(dataset, batch_size=1024, collate_fn=collate_fn,
                        num_workers=2, pin_memory=False)  #pin_mem=True neu GPU
    valid_loader = DataLoader(valid_dataset, batch_size=1024, collate_fn=collate_fn,
                              num_workers=2, pin_memory=False)
    
    model = SGNS(len(id2word), dim=200).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    model.train()
    lossplot = LossPlotter()
    for epoch in range(10):
        train_loss = 0.0
        for step, batch in enumerate(loader):
            center = batch["center"].to(device)
            pos = batch["pos"].to(device)
            neg = batch["neg"].to(device)
            
            loss = model(center, pos, neg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss += loss

            if step % 2000 == 0:
                print(f"step {step}: loss={loss.item():.4f}")
        train_loss = train_loss / len(loader)

        valid_loss = 0.0
        for batch in valid_loader:
            center = batch["center"]
            pos = batch["pos"]
            neg = batch["neg"]
            
            loss = model(center, pos, neg)
            valid_loss += loss
        valid_loss = valid_loss / len(valid_loader)
        evaluate(model, id2word, wid=["man", "woman", "king", "queen", "happy", "good", "bad", "nice", "time"])
        lossplot.update(epoch, train_loss, valid_loss)


        bundle = {
            "word2id" : word2id,
            "id2word" : id2word,
            "w_in" : model.embed_in.weight.detach().cpu().float(),
            "w_out" : model.embed_out.weight.detaach().cpu().float()
        }
        torch.save(bundle, f"/kaggle/working/w2v_epoch{epoch}.pt")







