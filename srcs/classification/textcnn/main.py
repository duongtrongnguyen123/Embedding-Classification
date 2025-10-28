import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))  # Thêm thư mục gốc Embedding-Classification/

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split

from srcs.classification.textcnn.data_utils import load_data
from srcs.classification.textcnn.dataset import ReviewDataset
from srcs.classification.textcnn.model import TextCNN
from srcs.classification.textcnn.train_eval import train_model

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent  # Cập nhật để trỏ đúng đến gốc
    DATA_DIR = BASE_DIR / "data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_data = torch.load(DATA_DIR / "vocab1.pt", map_location="cpu",weights_only=False)
    w_in, word2id = vocab_data["w_in"], vocab_data["word2id"]

    train_pos, y1 = load_data(DATA_DIR / "train" / "pos", 1)
    train_neg, y2 = load_data(DATA_DIR / "train" / "neg", 0)
    test_pos, y3 = load_data(DATA_DIR / "test" / "pos", 1)
    test_neg, y4 = load_data(DATA_DIR / "test" / "neg", 0)

    texts = train_pos + train_neg
    labels = y1 + y2
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.1, stratify=labels, random_state=42)

    train_ds = ReviewDataset(X_train, y_train, word2id)
    val_ds = ReviewDataset(X_val, y_val, word2id)
    test_ds = ReviewDataset(test_pos + test_neg, y3 + y4, word2id)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = TextCNN(w_in).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
                epochs=12, patience=3, save_path=BASE_DIR / "best_textcnn.pt")

if __name__ == "__main__":
    main()