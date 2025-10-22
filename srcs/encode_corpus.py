import torch 
import os 
import numpy as np

from data_pipe import iter_wiki_sentences

def pair_id_encode(a, b):
    return (a << 32 | b)
    





        


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    to_save_dir = os.path.join(cur_dir, "..", "data", "vocab.pt")
    vocab = torch.load(to_save_dir)
    o_word2id = vocab["o_word2id"]
    o_id2word = vocab["o_id2word"]
    old2new = vocab["old2new"]
    old2new_for_pair = vocab["old2new_for_pair"]
    word2id = vocab["word2id"]
    id2word = vocab["id2word"]
    counts = vocab["counts"]

    print(id2word[-5000:])

        
