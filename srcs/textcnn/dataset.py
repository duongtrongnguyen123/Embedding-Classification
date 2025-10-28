import torch
from torch.utils.data import Dataset
from src.data_utils import encode_text

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, word2id, max_len=300):
        self.X = [encode_text(t, word2id, max_len) for t in texts]
        self.y = labels

    def __len__(self): 
        return len(self.X)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.long), torch.tensor(self.y[i], dtype=torch.long)