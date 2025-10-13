import os
import pandas as pd
# Import hàm tải dữ liệu bạn đã có
from imdb_classifier import load_imdb_data_from_folders

# --- KHAI BÁO ĐƯỜNG DẪN (Giống trong imdb_classifier.py) ---
IMDB_DATA_DIR = os.path.join(os.getcwd(), 'data', 'aclImdb')
OUTPUT_CSV_FILE = os.path.join(os.getcwd(), 'data', 'imdb_full_reviews.csv')
# -----------------------------------------------------------

print("Bắt đầu tải 50.000 file TXT thô. Quá trình này sẽ mất 15-20 phút...")

# Tải dữ liệu từ các thư mục (BƯỚC CHẠY LÂU NHẤT)
df_full = load_imdb_data_from_folders(IMDB_DATA_DIR, splits=['train', 'test'])

print("Hoàn tất tải dữ liệu. Bắt đầu lưu trữ...")

# LƯU TRỮ VÀO MỘT FILE CSV DUY NHẤT
df_full.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')

print(f"Lưu trữ thành công vào: {OUTPUT_CSV_FILE}")
print("Bây giờ bạn có thể sửa code classifier để tải từ file CSV này.")