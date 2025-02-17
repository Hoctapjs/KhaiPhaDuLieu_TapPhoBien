import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

# Đọc dữ liệu từ file CSV
file_path = "Groceries_dataset.csv"  # Thay đổi đường dẫn nếu cần
df = pd.read_csv(file_path)

# Nhóm các sản phẩm theo Member_number (có thể đổi sang Date nếu cần)
transactions = df.groupby("Member_number")["itemDescription"].apply(list).tolist()

# Mã hóa dữ liệu thành ma trận 0-1
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Áp dụng thuật toán Apriori để tìm tập phổ biến
min_support = 0.01  # Thay đổi nếu cần
frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

# Tìm tập phổ biến tối đại
maximal_itemsets = frequent_itemsets[~frequent_itemsets['itemsets'].apply(lambda x: any(
    set(x).issubset(set(y)) for y in frequent_itemsets['itemsets'] if set(x) != set(y)))].copy()

# Tìm tập phổ biến đóng
closed_itemsets = frequent_itemsets[~frequent_itemsets['itemsets'].apply(lambda x: any(
    set(x).issubset(set(y)) and frequent_itemsets.loc[frequent_itemsets['itemsets'] == y, 'support'].values[0] == frequent_itemsets.loc[frequent_itemsets['itemsets'] == x, 'support'].values[0]
    for y in frequent_itemsets['itemsets'] if set(x) != set(y)))].copy() 

# Chuyển đổi frozenset thành danh sách chuỗi để dễ đọc trong file CSV
frequent_itemsets["itemsets"] = frequent_itemsets["itemsets"].apply(lambda x: ', '.join(x))
maximal_itemsets["itemsets"] = maximal_itemsets["itemsets"].apply(lambda x: ', '.join(x))
closed_itemsets["itemsets"] = closed_itemsets["itemsets"].apply(lambda x: ', '.join(x))

# Xuất ra file CSV
frequent_itemsets.to_csv("frequent_itemsets.csv", index=False)
maximal_itemsets.to_csv("maximal_itemsets.csv", index=False)
closed_itemsets.to_csv("closed_itemsets.csv", index=False)

# Hiển thị kết quả
print("Top Frequent Itemsets:")
print(frequent_itemsets.head())

print("\nTop Maximal Frequent Itemsets:")
print(maximal_itemsets.head())

print("\nTop Closed Frequent Itemsets:")
print(closed_itemsets.head())
