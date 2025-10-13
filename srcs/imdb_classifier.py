import pandas as pd
import numpy as np
import os
import torch 
import re
import unicodedata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from LogisticRegression import LogisticRegressionScratch, metrics
from text8_embedding import SGNS # Hoặc nơi nào chứa SGNS.normalize/cần thiết

# ==============================================================================
# PHẦN A: TIỀN XỬ LÝ ĐỒNG BỘ (CẦN SAO CHÉP TỪ CODE WORD2VEC GỐC CỦA BẠN)
# ==============================================================================

# Cần định nghĩa lại các hằng số Regex và Set bị thiếu từ code gốc của bạn
_tok_re = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*(?:\s?['’]\s?[A-Za-z]+)?|\d+|[.!?]")
_year_re = re.compile(r"\d{4}") # Ví dụ đơn giản cho regex năm
ROMAN_SMALL = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"} # Ví dụ 
END = {".", "!", "?"}

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return (s.replace("’", "'").replace("‘", "'")
             .replace("“", '"').replace("”", '"'))

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

def _norm_token(tok: str) -> str:
    # PHẢI ĐỒNG BỘ CÁC ĐỊNH NGHĨA ROMAN_SMALL, _year_re với code gốc!
    if tok.isdigit():
        if _year_re.fullmatch(tok):
            return "<year>"
        return "<digits>" if len(tok) <= 1 else "<nums>"
    if tok in ROMAN_SMALL:
        return "<century>"
    return tok

def preprocess_review(review_text: str):
    """Áp dụng quy trình tiền xử lý đồng bộ với dữ liệu Word Embedding (WikiText)."""
    processed_tokens = []
    
    for tok in tokenize(review_text):
        if tok in END:
            continue
            
        for st in expand_contraction(tok):
            normalized_st = _norm_token(st)
            
            # Chỉ giữ các token đã chuẩn hóa (isalpha() hoặc token đặc biệt)
            if normalized_st and (normalized_st.isalpha() or normalized_st in {"<year>", "<digit>", "<nums>", "<century>"}):
                processed_tokens.append(normalized_st)
                
    return processed_tokens


# ==============================================================================
# PHẦN B: TẢI DỮ LIỆU THÔ IMDB (Đã xác định từ cấu trúc folder)
# ==============================================================================

def load_imdb_data_from_folders(data_dir: str, splits=['train', 'test']) -> pd.DataFrame:
    """Tải dữ liệu IMDB từ cấu trúc thư mục (aclImdb/train/pos, neg...)."""
    all_data = []

    for split in splits:
        for sentiment, label in [('pos', 1), ('neg', 0)]:
            data_path = os.path.join(data_dir, split, sentiment)
            if not os.path.exists(data_path):
                 print(f"Cảnh báo: Không tìm thấy thư mục {data_path}")
                 continue

            for filename in os.listdir(data_path):
                if filename.endswith(".txt"):
                    file_path = os.path.join(data_path, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            review = f.read()
                        all_data.append({'review': review, 'sentiment': label, 'split': split})
                    except Exception as e:
                        # Bỏ qua nếu có lỗi đọc file
                        print(f"Bỏ qua file {file_path} do lỗi: {e}") 
                        
    return pd.DataFrame(all_data)


# ==============================================================================
# PHẦN C: TẢI VÀ CHUẨN BỊ WORD EMBEDDING
# ==============================================================================

def load_and_prepare_embedding(embed_file_path):
    """Tải ma trận WE, chuẩn hóa và trả về dưới dạng NumPy."""
    try:
        embed = torch.load(embed_file_path, map_location='cpu')
    except FileNotFoundError:
        raise FileNotFoundError(f"Lỗi: Không tìm thấy file WE tại {embed_file_path}. Vui lòng kiểm tra lại đường dẫn.")
        
    w_in = embed['w_in']
    w_out = embed['w_out']
    word2id = embed["word2id"]

    # 1. Kết hợp W_in và W_out (Average)
    combined_embedding = (w_in + w_out) * 0.5 
    
    # 2. Chuyển sang NumPy (float64 là tốt nhất cho các phép tính)
    WE_numpy = combined_embedding.detach().cpu().numpy().astype(np.float64)

    # 3. Chuẩn hóa L2 (rất quan trọng)
    WE_final = normalize(WE_numpy, axis=1)

    unk_id = word2id.get("<unk>", 0)
    
    return WE_final, word2id, unk_id


# ==============================================================================
# PHẦN D: TRÍCH XUẤT ĐẶC TRƯNG (MAPPING & GAP)
# ==============================================================================

def tokenize_and_map(review_text: str, word2id: dict, unk_id: int) -> np.ndarray:
    """Áp dụng tiền xử lý đồng bộ và ánh xạ các token sang ID số."""
    # Dùng hàm tiền xử lý đã đồng bộ
    processed_tokens = preprocess_review(review_text)
    
    # Ánh xạ từ token sang ID, dùng UNK_ID nếu từ không có trong từ điển
    ids = [word2id.get(token, unk_id) for token in processed_tokens]
    
    return np.array(ids, dtype=np.int64)

def apply_gap(ids_array: np.ndarray, WE_matrix: np.ndarray) -> np.ndarray:
    """Thực hiện Global Average Pooling (GAP) để tạo Review Vector."""
    
    vectors = WE_matrix[ids_array]
    
    if len(vectors) == 0:
        return np.zeros(WE_matrix.shape[1], dtype=np.float64) 
    
    # Tính trung bình cộng theo trục 0 (các hàng/vector từ)
    review_vector = np.mean(vectors, axis=0)
    
    return review_vector


# ==============================================================================
# PHẦN E: CHƯƠNG TRÌNH CHÍNH
# ==============================================================================

if __name__ == "__main__":
    
    # ------------------ KHAI BÁO ĐƯỜNG DẪN ------------------
    # ⚠️ THAY THẾ ĐƯỜNG DẪN THỰC TẾ CỦA BẠN ⚠️
    WE_FILE_PATH = os.path.join(os.getcwd(), 'vocab.pt', 'vocab.pt') 
    CACHED_CSV_FILE = os.path.join(os.getcwd(), 'data', 'imdb_full_reviews.csv')
    # --------------------------------------------------------
    
    # 1. Tải và Chuẩn bị Word Embedding
    print(f"1. Đang tải Word Embedding từ: {WE_FILE_PATH}")
    WE_MATRIX, WORD2ID, UNK_ID = load_and_prepare_embedding(WE_FILE_PATH)
    D_DIM = WE_MATRIX.shape[1]
    print(f"   -> Tải thành công. Kích thước WE: {WE_MATRIX.shape}")
    
    # 2. Tải Dữ liệu IMDB
    print(f"2. Đang tải dữ liệu từ file cached: {CACHED_CSV_FILE}")
    
    try:
        # Tải nhanh từ CSV
        df_all = pd.read_csv(CACHED_CSV_FILE, encoding='utf-8')
    except FileNotFoundError:
        print("LỖI: Không tìm thấy file cached! Vui lòng chạy script save_data.py trước!")
        exit()
    
    # Chia dữ liệu theo cột 'split' đã lưu
    df_train = df_all[df_all['split'] == 'train'].reset_index(drop=True)
    df_test = df_all[df_all['split'] == 'test'].reset_index(drop=True)

    # 3. Áp dụng Tiền xử lý và GAP cho Tập Huấn luyện
    print("3. Áp dụng tiền xử lý và GAP cho tập huấn luyện...")
    X_train_ids = df_train['review'].apply(tokenize_and_map, args=(WORD2ID, UNK_ID)).values
    X_train_features = np.array([apply_gap(ids, WE_MATRIX) for ids in X_train_ids], dtype=np.float64)
    y_train = df_train['sentiment'].values
    
    # 4. Áp dụng Tiền xử lý và GAP cho Tập Kiểm tra
    print("4. Áp dụng tiền xử lý và GAP cho tập kiểm tra...")
    X_test_ids = df_test['review'].apply(tokenize_and_map, args=(WORD2ID, UNK_ID)).values
    X_test_features = np.array([apply_gap(ids, WE_MATRIX) for ids in X_test_ids], dtype=np.float64)
    y_test = df_test['sentiment'].values
    
    print(f"   -> Kích thước Feature Vector: {X_train_features.shape[0]}x{D_DIM}")
    
    # 5. Huấn luyện Logistic Regression
    print("5. Bắt đầu huấn luyện Logistic Regression...")
    clf = LogisticRegressionScratch(
        n_iters=3000, 
        lr=0.5, 
        reg_lambda=0.01, # L2 Regularization để chống overfit
        batch_size=128,
        val_ratio=0.05,
        early_stopping=True,
        patience=50
    )
    
    clf.fit(X_train_features, y_train)

    # 6. Đánh giá
    print("6. Đánh giá mô hình trên tập kiểm tra...")
    y_pred = clf.predict(X_test_features)
    acc, prec, rec, f1 = metrics(y_test, y_pred)
    
    print("\n==================================")
    print("KẾT QUẢ PHÂN LOẠI REVIEW PHIM (GAP + LR)")
    print(f"Accuracy (Độ chính xác) = {acc:.4f}")
    print(f"Precision (Độ chuẩn xác) = {prec:.4f}")
    print(f"Recall (Độ phủ) = {rec:.4f}")
    print(f"F1-Score = {f1:.4f}")
    print("==================================")