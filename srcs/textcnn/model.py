import torch
import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self, embedding_matrix, num_classes=2, kernel_sizes=(2,3,4,5), num_filters=128, dropout=0.5):
        super().__init__()
        vocab_size, emb_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix.detach().clone().float(), freeze=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=emb_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_filters) for _ in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (B, emb_dim, seq_len)
        outs = []
        for conv, bn in zip(self.convs, self.bns):
            o = torch.relu(bn(conv(x)))
            pooled = torch.max(o, dim=2)[0]
            outs.append(pooled)
        x = torch.cat(outs, dim=1)
        x = self.dropout(x)
        return self.fc(x)