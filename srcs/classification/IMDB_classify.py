import os
import re
import unicodedata
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

# TOKENIZE / PREPROCESS
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

def preprocess_imdb(text: str):
    toks = []
    for tok in tokenize(text):
        toks.extend(expand_contraction(tok))
    return toks


AUX = {"do","does","did","am","is","are","was","were","be","been","being",
       "have","has","had","will","would","shall","should","can","could",
       "may","might","must"}
INTENS = {"really","very","quite","so","too","extremely","fairly",
          "pretty","rather","somewhat","kinda","sorta","at","all"}

def aux_intens_old_ids(word2id):
    s = set()
    for w in (AUX | INTENS):
        if w in word2id:
            s.add(int(word2id[w]))
    return s


def encode_sentence_synced(tokens, word2id, old2new_nd, old2new_for_pair,
                           skip_old_ids, not_old_id,
                           max_len=200, pad_new_id=0):
    """
    - Token không có trong word2id (OLD) => drop
    - Nếu prev = 'not' và current ∈ (AUX ∪ INTENS) => skip current, giữ 'not'
    - Ưu tiên merge cặp: (prev_old<<32)|cur_old -> new_pair_id
    - Nếu không merge được, map old->new qua old2new_nd (np.ndarray; -1 => drop)
    - Pad/truncate bằng pad_new_id (NEW)
    """
    out = []

    prev_old = None   
    prev_new = None  
    prev_in_negate = False

    def flush_prev():
        nonlocal prev_old, prev_new, prev_in_negate
        if prev_new is not None:
            out.append(prev_new)
        prev_old = None
        prev_new = None
        prev_in_negate = False

    for tok in tokens:
        cur_old = word2id.get(tok, None)

        if cur_old is None:
            if not prev_in_negate:
                flush_prev()
            continue

        if prev_old is not None and prev_in_negate and (cur_old in skip_old_ids):
            continue

        if prev_old is not None:
            
            key = int((np.int64(prev_old) << 32) | np.int64(cur_old))
            pair_new = old2new_for_pair.get(key, None)
            if pair_new is not None:
                out.append(int(pair_new))
                prev_old = None
                prev_new = None
                prev_in_negate = False
                continue
            else:
                
                flush_prev()

       
        if 0 <= cur_old < old2new_nd.size:
            new_id = int(old2new_nd[cur_old])
        else:
            new_id = -1

        if new_id < 0:
      
            if not prev_in_negate:
                flush_prev()
            continue

        
        prev_old = cur_old
        prev_new = new_id
        prev_in_negate = (not_old_id is not None and prev_old == not_old_id)

   
    flush_prev()

    
    if len(out) < max_len:
        out += [pad_new_id] * (max_len - len(out))
    else:
        out = out[:max_len]
    return out


#  DATA LOADING 
def load_folder_as_texts(folder, label):
    texts, labels = [], []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(label)
    return texts, labels


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, word2id, old2new_nd, old2new_for_pair,
                 max_len=200, pad_new_id=0):
        assert isinstance(old2new_nd, np.ndarray), "old2new phải là np.ndarray"

        skip_ids = aux_intens_old_ids(word2id)
        not_old_id = word2id.get("not", None)

        self.X = []
        for t in texts:
            toks = preprocess_imdb(t)
            ids = encode_sentence_synced(
                toks, word2id, old2new_nd, old2new_for_pair,
                skip_ids, not_old_id, max_len=max_len, pad_new_id=pad_new_id
            )
            self.X.append(ids)
        self.y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.long), torch.tensor(self.y[i], dtype=torch.long)
    
class PreprocessedReviewDataset(Dataset):
    """Một Dataset SIÊU NHANH, chỉ tải dữ liệu đã được xử lý (cached)."""
    def __init__(self, X_data_ids, y_data_labels):
        self.X = X_data_ids   
        self.y = y_data_labels 
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        
        return torch.tensor(self.X[i], dtype=torch.long), torch.tensor(self.y[i], dtype=torch.long)


# MODEL: LOGISTIC REGRESSION 
class LogisticRegressionOnMean(nn.Module):
    """Mean pooling (bỏ pad) rồi Linear -> logits."""
    def __init__(self, embedding_matrix, num_classes=2, padding_idx=0, dropout_p=0.0, freeze_embed=False):
        super().__init__()
        vocab_size, emb_dim = embedding_matrix.shape
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix.detach().clone().float(),
            freeze=freeze_embed,
            padding_idx=padding_idx
        )
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.linear = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)                             # [B, L, D]
        mask = (x != self.padding_idx).unsqueeze(-1)        # [B, L, 1]
        emb = emb * mask
        sum_vec = emb.sum(dim=1)                            # [B, D]
        cnt = mask.sum(dim=1).clamp(min=1)                  # [B, 1]
        mean_vec = sum_vec / cnt
        return self.linear(self.dropout(mean_vec))          # [B, C]

# CÁC HÀM HUẤN LUYỆN VÀ ĐÁNH GIÁ
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.append(preds.cpu())
            all_labels.append(y_batch.cpu())
            
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    f1 = f1_score(all_labels, all_preds, average='binary') 
    acc = (all_preds == all_labels).mean()
    return acc, f1

# MAIN 
def main():
    # 1.Tải Artifacts 
    print("1. Đang tải Artifacts (WE, Dicts, Mappings)...")
    
    vocab_path = "data/vocab.pt"
    embed_in_path = "data/embed_in.pt"
    embed_out_path = "data/embed_out.pt"
    
    try:
        
        vocab_data = torch.load(vocab_path, map_location="cpu", weights_only=False)
        embed_in = torch.load(embed_in_path, map_location="cpu", weights_only=False)
        embed_out = torch.load(embed_out_path, map_location="cpu", weights_only=False)
    except FileNotFoundError as e:
        print(f"LỖI: Không tìm thấy file. Vui lòng kiểm tra các đường dẫn trong thư mục 'data'.")
        print(f"Lỗi chi tiết: {e}")
        return

    # Tính toán ma trận embedding cuối cùng
    w = (embed_in + embed_out) / 2  

    w_centered = w - w.mean(dim=0, keepdim=True)


    u, s, v = torch.svd(w_centered)   # hoặc torch.linalg.svd
    first = v[:, 0]


    w = w_centered - (w_centered @ first.unsqueeze(1)) * first.unsqueeze(0)
    
    # Tải các artifacts từ file vocab
    word2id = vocab_data["word2id"]             # dict token -> OLD id
    old2new_nd = vocab_data["old2new"]          # np.ndarray OLD -> NEW (=-1 nếu drop)
    old2new_for_pair = vocab_data["old2new_for_pair"] # dict key(old-old) -> NEW id

    # Tìm NEW pad id 
    old_pad = word2id.get("<pad>", 0) 
    pad_new_id = 0 
    
    if isinstance(old2new_nd, torch.Tensor):
        old2new_nd = old2new_nd.cpu().numpy()

    if isinstance(old2new_nd, np.ndarray) and 0 <= old_pad < old2new_nd.size:
        tmp = int(old2new_nd[old_pad])
        if 0 <= tmp < w.shape[0]:
            pad_new_id = tmp
    print(f"   -> Đã tải thành công. NEW padding ID được đặt là: {pad_new_id}")

    #2. Tải IMDB data
   
    cache_path = "data/imdb_processed_cache.pt"
    base_dir = "data"
    max_len = 256 

    if os.path.exists(cache_path):
        print(f"2. Đang tải dữ liệu ĐÃ XỬ LÝ: {cache_path}")
        cached_data = torch.load(cache_path)
        train_ds = PreprocessedReviewDataset(cached_data['train_x'], cached_data['train_y'])
        test_ds = PreprocessedReviewDataset(cached_data['test_x'], cached_data['test_y'])
        
    else:
        print(f"2. Không tìm thấy cache. Đang xử lý 50,000 file...")
        train_pos_path = os.path.join(base_dir, "train", "pos")
        train_neg_path = os.path.join(base_dir, "train", "neg")
        test_pos_path = os.path.join(base_dir, "test", "pos")
        test_neg_path = os.path.join(base_dir, "test", "neg")

        try:
            train_pos, y1 = load_folder_as_texts(train_pos_path, 1)
            train_neg, y2 = load_folder_as_texts(train_neg_path, 0)
            test_pos,  y3 = load_folder_as_texts(test_pos_path, 1)
            test_neg,  y4 = load_folder_as_texts(test_neg_path, 0)
        except FileNotFoundError:
            print(f"LỖI: Không tìm thấy thư mục dữ liệu IMDB tại {base_dir}")
            return

        train_texts = train_pos + train_neg
        train_labels = y1 + y2
        test_texts  = test_pos  + test_neg
        test_labels = y3 + y4

        # Tạo Dataset 
        print("   -> Đang tạo tập huấn luyện")
        train_ds_processing = ReviewDataset(train_texts, train_labels, word2id, old2new_nd, old2new_for_pair,
                                     max_len=max_len, pad_new_id=pad_new_id)
        print("   -> Đang tạo tập kiểm tra")
        test_ds_processing = ReviewDataset(test_texts,  test_labels,  word2id, old2new_nd, old2new_for_pair,
                                     max_len=max_len, pad_new_id=pad_new_id)
        
        # Lưu vào Cache
        print(f"   -> Đang lưu cache vào {cache_path}")
        torch.save({
            'train_x': train_ds_processing.X, # Lưu list các list ID
            'train_y': train_ds_processing.y, # Lưu list các nhãn
            'test_x': test_ds_processing.X,
            'test_y': test_ds_processing.y,
            'pad_new_id': pad_new_id # Lưu cả pad_id
        }, cache_path)
        
        # Dùng PreprocessedReviewDataset cho lần chạy này
        train_ds = PreprocessedReviewDataset(train_ds_processing.X, train_ds_processing.y)
        test_ds = PreprocessedReviewDataset(test_ds_processing.X, test_ds_processing.y)

    # 3. Dataloader 
    print("3. Đang tạo Dataloader")
    
    batch_size = 128
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    # 4. Model / Train / Eval 
    print("4. Đang khởi tạo mô hình...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   -> Sử dụng thiết bị: {device}")

    model = LogisticRegressionOnMean(
        w, 
        num_classes=2, 
        padding_idx=pad_new_id,
        dropout_p=0.4,          
        freeze_embed=False     
    )
    model.to(device)

    #Cài đặt Huấn luyện
    N_EPOCHS = 40

    learning_rate = 1e-4 
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"5. Bắt đầu huấn luyện {N_EPOCHS} epochs (LR={learning_rate})...")
    for epoch in range(1, N_EPOCHS + 1):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_acc, test_f1 = evaluate(model, test_loader, device)
        
        end_time = time.time()
        print(f"--- Epoch {epoch}/{N_EPOCHS} ---")
        print(f"   Thời gian: {end_time - start_time:.2f}s")
        print(f"   Loss Huấn luyện: {train_loss:.4f}")
        print(f"   Độ chính xác (Test): {test_acc:.4f}")
        print(f"   F1-Score (Test): {test_f1:.4f}")

if __name__ == "__main__":
    main() 
