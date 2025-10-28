import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(BASE_DIR))

from srcs.classification.textcnn.data_utils import load_data
from srcs.classification.textcnn.dataset import ReviewDataset
from srcs.classification.textcnn.model import TextCNN
from srcs.classification.textcnn.train_eval import train_model


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================================
    # 1️⃣ Tải Artifacts (embedding, vocab, mapping)
    # ==========================================================
    print("1. Đang tải Artifacts (WE, Dicts, Mappings)...")

    vocab_path = DATA_DIR / "vocab.pt"
    embed_in_path = DATA_DIR / "embed_in.pt"
    embed_out_path = DATA_DIR / "embed_out.pt"

    try:
        vocab_data = torch.load(vocab_path, map_location="cpu", weights_only=False)
        embed_in = torch.load(embed_in_path, map_location="cpu", weights_only=False)
        embed_out = torch.load(embed_out_path, map_location="cpu", weights_only=False)
    except FileNotFoundError as e:
        print(f"LỖI: Không tìm thấy file. Vui lòng kiểm tra các đường dẫn trong thư mục 'data'.")
        print(f"Lỗi chi tiết: {e}")
        return

    # ==========================================================
    # 2️⃣ Tính toán embedding trung bình + chuẩn hoá
    # ==========================================================
    w = (embed_in + embed_out) / 2
    w_centered = w - w.mean(dim=0, keepdim=True)

    # Loại bỏ thành phần chính đầu tiên
    u, s, v = torch.svd(w_centered)
    first = v[:, 0]
    w = w_centered - (w_centered @ first.unsqueeze(1)) * first.unsqueeze(0)

    # ==========================================================
    # 3️⃣ Xử lý mapping vocab và pad id
    # ==========================================================
    word2id = vocab_data["word2id"]                      # dict token -> OLD id
    old2new_nd = vocab_data["old2new"]                   # np.ndarray OLD -> NEW (=-1 nếu drop)
    old2new_for_pair = vocab_data["old2new_for_pair"]    # dict key(old-old) -> NEW id

    old_pad = word2id.get("<pad>", 0)
    pad_new_id = 0

    if isinstance(old2new_nd, torch.Tensor):
        old2new_nd = old2new_nd.cpu().numpy()

    if isinstance(old2new_nd, np.ndarray) and 0 <= old_pad < old2new_nd.size:
        tmp = int(old2new_nd[old_pad])
        if 0 <= tmp < w.shape[0]:
            pad_new_id = tmp

    print(f"   -> Đã tải thành công. NEW padding ID được đặt là: {pad_new_id}")

    # ==========================================================
    # 4️⃣ Tải dữ liệu train / test
    # ==========================================================
    train_pos, y1 = load_data(DATA_DIR / "train" / "pos", 1)
    train_neg, y2 = load_data(DATA_DIR / "train" / "neg", 0)
    test_pos, y3 = load_data(DATA_DIR / "test" / "pos", 1)
    test_neg, y4 = load_data(DATA_DIR / "test" / "neg", 0)

    texts = train_pos + train_neg
    labels = y1 + y2
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.1, stratify=labels, random_state=42
    )

    train_ds = ReviewDataset(X_train, y_train, word2id)
    val_ds = ReviewDataset(X_val, y_val, word2id)
    test_ds = ReviewDataset(test_pos + test_neg, y3 + y4, word2id)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # ==========================================================
    # 5️⃣ Huấn luyện mô hình
    # ==========================================================
    model = TextCNN(w).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
                epochs=12, patience=3, save_path=BASE_DIR / "best_textcnn.pt")


if __name__ == "__main__":
    main()
