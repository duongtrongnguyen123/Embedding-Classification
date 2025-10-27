import os
import multiprocessing as mp
import random, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import time

from data_pipeline.data_pipe_ids import SkipGramPairIterable



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
        neg = self.neg_sample(centers.shape[0], generator)
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


def evaluate(model: SGNS, word2id, id2word, wid=["happy", "good", "bad", "shit", "excellent", "outstanding", "masterpiece", "delightful", "awful", "terrible", "boring", "cringe", "waste"]):
    ids = [word2id[i] for i in wid]
    cos, idx = model.most_similar(ids, 4)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_dir = os.path.join(base_dir, "..", "..", "data", "vocab.pt")
    train_corpus_dir = os.path.join(base_dir, "..", "..", "data", "n_train_corpus.bin")
    valid_corpus_dir = os.path.join(base_dir, "..", "..", "data", "n_valid_corpus.bin")
    vocab = torch.load(vocab_dir, map_location='cpu')
    print("loading vocab...")
    word2id = vocab["word2id"]
    id2word = vocab["id2word"]
    counts = vocab["counts"]
    keep_probs = vocab["keep_probs"]
    keep_probs[word2id["abd"]] = 0.001
    keep_probs[word2id["yuk"]] = 0.001
    print("loaded vocab...")
    model = SGNS(len(id2word), neg_k=15, counts= counts, dim=256).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,        
        betas=(0.9, 0.99),
        eps=1e-8,         
        weight_decay=1e-5  
)

    lossplot = LossPlotter()


    BATCH_SIZE = 32768
    train_pair_iterable = SkipGramPairIterable(path=train_corpus_dir, train=True, keep_probs=keep_probs,
                                               window=5, batch_size=BATCH_SIZE, seed=121,
                                               device=device)
    valid_pair_iterable = SkipGramPairIterable(path=valid_corpus_dir, train=False, keep_probs=keep_probs, 
                                               window=5, batch_size=BATCH_SIZE, seed=124,    
                                               device=device)

   
    train_loader = DataLoader(train_pair_iterable, batch_size=None, 
                              num_workers=8,      
                              prefetch_factor=8, persistent_workers=True, 
                              pin_memory=False)

    valid_loader = DataLoader(valid_pair_iterable, batch_size=None,
                              num_workers=8,      
                              prefetch_factor=8, persistent_workers=True, 
                              pin_memory=False)

    start = time.time()
    base_seed = 111

    for epoch in range(10):
        model.train()
        generator = torch.Generator(device=device).manual_seed(epoch+base_seed)
        train_loss = 0.0
        total_sample = 0
        for step, (center,context) in enumerate(train_loader):
            B = center.shape[0]
            opt.zero_grad(set_to_none=True)
            loss = model(center, context, generator)
            loss.backward()
            opt.step()
            train_loss += loss * B
            total_sample += B

            if step % 200 == 0:
                stop = time.time()
                elapsed = stop - start
                prev_step = step-200
                print(f"step  {prev_step}->{step}: {elapsed}")
                start = stop

        print(f"total sample: {total_sample}")
        train_loss = train_loss / total_sample

    
        model.eval()
        valid_loss = 0.0
        total_sample = 0
        for step, (center, context) in enumerate(valid_loader):
            B = center.shape[0]
            loss = model(center, context, generator)
            opt.step()
            valid_loss += loss * B
            total_sample += B


        valid_loss = valid_loss / total_sample
        evaluate(model, word2id, id2word)
        print(f"diff: {train_loss-valid_loss}")
        #lossplot.update(epoch, train_loss, valid_loss)
