import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split
from pathlib import Path

from src.data_utils import load_data
from src.dataset import ReviewDataset
from src.model import TextCNN
from src.train_eval import train_model

def main():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocab
    vocab_data = torch.load(DATA_DIR / "vocab.pt", map_location="cpu")
    w_in, word2id = vocab_data["w_in"], vocab_data["word2id"]

    # Load dữ liệu từ data/train/pos, data/train/neg, data/test/pos, data/test/neg
    train_pos, y1 = load_data(DATA_DIR / "train" / "pos", 1)
    train_neg, y2 = load_data(DATA_DIR / "train" / "neg", 0)
    test_pos,  y3 = load_data(DATA_DIR / "test" / "pos", 1)
    test_neg,  y4 = load_data(DATA_DIR / "test" / "neg", 0)

    # Gộp train để chia train/val
    texts = train_pos + train_neg
    labels = y1 + y2
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.1, stratify=labels, random_state=42
    )

    # Dataset
    train_ds = ReviewDataset(X_train, y_train, word2id)
    val_ds   = ReviewDataset(X_val, y_val, word2id)
    test_ds  = ReviewDataset(test_pos + test_neg, y3 + y4, word2id)

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

    # Model
    model = TextCNN(w_in).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

    # Train
    train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        epochs=12, patience=3, save_path=BASE_DIR / "best_textcnn.pt"
    )

if __name__ == "__main__":
    main()