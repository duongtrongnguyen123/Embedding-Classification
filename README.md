```
Embedding-Classification/
└─ srcs/
   └─ word2vec/
      ├─ data_pipeline/              # Xử lý dữ liệu thô → vocab → encode → memmap
      │  ├─ _count_fast.pyx         # Cython đếm token nhanh (first pass)
      │  ├─ _encode_corpus.pyx      # Cython encode corpus sang ID (second pass)
      │  ├─ build_vocab.py          # Tạo vocab, tính keep_prob, subsampling
      │  ├─ count_tokens.py         # Script đếm tần suất từ
      │  ├─ data_pipe.py            # Iterator cho text thô
      │  ├─ data_pipe_ids.py        # Iterator cho dữ liệu đã encode ID
      │  ├─ encode_corpus.py        # Encode corpus dùng vocab → lưu memmap
      │  ├─ review_dataset_iter.py  # Iterator cho tập review (IMDB/Amazon)
      │  └─ setup.py                # Build Cython extensions
      │
      ├─ embedding/                  # Huấn luyện Word2Vec trên dữ liệu ID
      │  ├─ embedding_ids.py        # Training loop SGNS/CBOW chính
      │  ├─ text8_embedding.py      # Ví dụ train trên text8
      │  └─ wikitext_embedding.py   # Ví dụ train trên WikiText
      │
      ├─ notebook/                   # Notebook minh hoạ / thử nghiệm
      │  ├─ train_reviews_ids.ipynb # Train Word2Vec trên review corpus
      │  └─ train_wikitext.ipynb    # Train Word2Vec trên WikiText
      │
      └─ test/                       # Unit test & benchmark
         ├─ speedtest.py            # Benchmark tốc độ nhân ma trận
         ├─ test_encode.py          # Test encode corpus → ID
         └─ test_fast_count.py      # Test module đếm nhanh _count_fast.pyx
```
