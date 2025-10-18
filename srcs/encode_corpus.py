import torch 
import os 
import numpy as np




if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    to_save_dir = os.path.join(cur_dir, "..", "data", "vocab.pt")

    vocab = torch.load(to_save_dir)

    word2id = vocab["word2id"]
    k = 0
    for i, j in word2id.items():
        print(f"{i}->{j}")
        k += 1
        if k >= 50:
            break
