# Phân tích Cảm xúc Phim IMDB: Pipeline PyTorch với Embedding Đồng bộ và Fine-tuning

Dự án này triển khai mô hình phân loại cảm xúc nhị phân (Tích cực/Tiêu cực) trên bộ dữ liệu review phim IMDB, tập trung vào việc đồng bộ hóa chặt chẽ quy trình tiền xử lý và chiến lược tinh chỉnh (fine-tuning) Word Embedding.

## Cài đặt và Sử dụng

### Yêu cầu

* Python 3.x
* PyTorch
* NumPy
* Scikit-learn
* spaCy (`pip install spacy`)
* Mô hình spaCy tiếng Anh (`python -m spacy download en_core_web_sm`)

### Dữ liệu

1.  **IMDB Dataset:** Đặt thư mục `aclImdb` (chứa `train/pos`, `train/neg`, `test/pos`, `test/neg`) vào thư mục `data/`.
2.  **Artifacts Embedding:** Đặt các file `vocab.pt`, `embed_in.pt`, `embed_out.pt` vào thư mục `data/`.

## Cấu trúc
```
Embedding-Classification/
└─ srcs/
   ├─ word2vec/
   |   ├─ data_pipeline/             # Xử lý dữ liệu thô → vocab → encode → memmap
   |   │  ├─ _count_fast.pyx         # Cython đếm token nhanh (first pass)
   |   │  ├─ _encode_corpus.pyx      # Cython encode corpus sang ID (second pass)
   |   │  ├─ build_vocab.py          # Tạo vocab, tính keep_prob, subsampling
   |   │  ├─ count_tokens.py         # Script đếm tần suất từ
   |   │  ├─ data_pipe.py            # Iterator cho text thô
   |   │  ├─ data_pipe_ids.py        # Iterator cho dữ liệu đã encode ID
   |   │  ├─ encode_corpus.py        # Encode corpus dùng vocab → lưu memmap
   |   │  ├─ review_dataset_iter.py  # Iterator cho tập review (IMDB/Amazon)
   |   │  └─ setup.py                # Build Cython extensions
   |   │
   |   ├─ embedding/                 # Huấn luyện Word2Vec trên dữ liệu ID
   |   │  ├─ embedding_ids.py        # Training loop SGNS/CBOW chính
   |   │  ├─ text8_embedding.py      # Ví dụ train trên text8
   |   │  └─ wikitext_embedding.py   # Ví dụ train trên WikiText
   |   │
   |   ├─ notebook/                  # Notebook minh hoạ / thử nghiệm
   |   │  ├─ train_reviews_ids.ipynb # Train Word2Vec trên review corpus
   |   │  └─ train_wikitext.ipynb    # Train Word2Vec trên WikiText
   |   │
   |   └─ test/                      # Unit test & benchmark
   |      ├─ speedtest.py            # Benchmark tốc độ nhân ma trận
   |      ├─ test_encode.py          # Test encode corpus → ID
   |      └─ test_fast_count.py      # Test module đếm nhanh _count_fast.pyx
   |
   └─ classification/
         ├─ IMDB_classify.py         # Phân loại bằng Logistic Regression + Fine-tuning
         └─ textcnn/                 # Phân loại bằng CNN model
            ├── data_utils.py # Hàm xử lý dữ liệu
            ├── dataset.py # Tạo dataset cho mô hình
            ├── main.py # Chạy huấn luyện và đánh giá
            ├── model.py # Định nghĩa mô hình TextCNN
            └─  train_eval.py # Hàm train và evaluate mô hình
```
