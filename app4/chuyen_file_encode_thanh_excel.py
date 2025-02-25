import pandas as pd

# Đường dẫn file CSV
csv_file = "df_encoded.csv"

# Đường dẫn file Excel đầu ra
excel_file = "df_encoded_excel.xlsx"

# Đọc file CSV
df = pd.read_csv(csv_file)

# Ghi dữ liệu vào file Excel
df.to_excel(excel_file, index=False)

print(f"File đã được chuyển đổi thành công: {excel_file}")
