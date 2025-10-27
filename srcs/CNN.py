import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
import unicodedata
from sklearn.metrics import f1_score

# ===================== TIỀN XỬ LÝ =========================
_tok_re = re.compile(
    r"[A-Za-z]+(?:-[A-Za-z]+)*(?:\s?['’]\s?[A-Za-z]+)?|\d+|[.!?]"
)

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return (s.replace("’", "'")
             .replace("‘", "'")
             .replace("“", '"')
             .replace("”", '"'))

def tokenize(s: str):
    if not s:
        return []
    s = normalize_text(s)
    return _tok_re.findall(s.lower())

def expand_contraction(tok: str):
    t = tok.replace(" ", "").replace("’", "'")
    if t.endswith("n't") and len(t) > 3: return [t[:-3], "not"]
    if t.endswith("'re") and len(t) > 3: return [t[:-3], "are"]
    if t.endswith("'ll") and len(t) > 3: return [t[:-3], "will"]
    if t.endswith("'ve") and len(t) > 3: return [t[:-3], "have"]
    if t.endswith("'m")  and len(t) > 2: return [t[:-2], "am"]
    if t.endswith("'d")  and len(t) > 2: return [t[:-2], "would"]
    if t.endswith("'s")  and len(t) > 2: return [t[:-2]]
    return [t]

def preprocess_imdb(text):
    tokens = []
    for tok in tokenize(text):
        for st in expand_contraction(tok):
            tokens.append(st)
    return tokens


# ===================== BỔ SUNG: HÀM ENCODE CORPUS =========================
def encode_corpus(old2new, negate_id, skip_id, old2new_for_pair, corpus, id2word):
    """
    Hàm Python mô phỏng _encode_corpus trong Cython.
    Dùng để mã hóa lại câu theo ánh xạ vocab và cặp từ (nếu cần).
    """
    new_corpus = []
    for sent in corpus:
        new_sent = []
        prev_id = None
        prev_in_negate = False

        for o_id in sent:
            # Bỏ qua token không hợp lệ
            if o_id not in old2new or old2new[o_id] < 0:
                if not prev_in_negate:
                    prev_id = None
                continue

            n_id = old2new[o_id]

            if prev_id is not None:
                prev_in_negate = prev_id in negate_id
                if prev_in_negate and o_id in skip_id:
                    continue

                # Mã hóa cặp từ (pair ID)
                o_pair_id = (prev_id << 32) | o_id
                if o_pair_id in old2new_for_pair:
                    if new_sent:
                        new_sent.pop()
                    new_sent.append(old2new_for_pair[o_pair_id])
                    prev_id = None
                    prev_in_negate = False
                    continue
                else:
                    new_sent.append(n_id)
            else:
                new_sent.append(n_id)

            prev_id = o_id
        new_corpus.append(new_sent)

    return new_corpus


############################
# 1. Dataset Reader
############################
def load_data(folder, label):
    texts, labels = [], []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(label)
    return texts, labels


############################
# 2. Encode text to IDs
############################
def encode_text(text, word2id, max_len=200):
    tokens = preprocess_imdb(text)
    ids = [word2id.get(tok, word2id.get("<unk>", 0)) for tok in tokens]

    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, word2id, max_len=200):
        self.X = [encode_text(t, word2id, max_len) for t in texts]
        self.y = labels
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.long), torch.tensor(self.y[i], dtype=torch.long)


############################
# 3. CNN Model
############################
class TextCNN(nn.Module):
    def __init__(self, embedding_matrix, num_classes=2):
        super().__init__()
        vocab_size, emb_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix.detach().clone().float(),
            freeze=False
        )
        self.conv = nn.Conv1d(in_channels=emb_dim, out_channels=100, kernel_size=3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.embedding(x)         # (B, L, D)
        x = x.permute(0, 2, 1)        # (B, D, L)
        x = self.conv(x)              # (B, 100, L-3+1)
        x = self.pool(x).squeeze(-1)  # (B, 100)
        return self.fc(x)


############################
# 4. MAIN FUNCTION
############################
def main():
    # ============ Load vocab & embedding ============
    vocab_data = torch.load("../data/vocab.pt", map_location="cpu")
    w_in = vocab_data["w_in"]
    word2id = vocab_data["word2id"]

    # ============ Load train & test data ============
    train_pos, y1 = load_data("../data/pos_train", 1)
    train_neg, y2 = load_data("../data/neg_train", 0)
    test_pos,  y3 = load_data("../data/pos_test", 1)
    test_neg,  y4 = load_data("../data/neg_test", 0)

    train_texts = train_pos + train_neg
    train_labels = y1 + y2
    test_texts = test_pos + test_neg
    test_labels = y3 + y4

    train_ds = ReviewDataset(train_texts, train_labels, word2id, max_len=200)
    test_ds = ReviewDataset(test_texts, test_labels, word2id, max_len=200)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # ============ Model ============
    model = TextCNN(w_in, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ============ Train ============
    for epoch in range(3):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")

    # ============ Evaluate ============
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average='binary')
    print(f"Test Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")


if __name__ == "__main__":
    main()
