import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

# Hàm chọn file CSV
def select_csv_file():
    root = Tk()
    root.withdraw()  # Ẩn cửa sổ Tkinter chính
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    return file_path

# Hàm xử lý dữ liệu từ file CSV
def process_data(file_path, min_support=0.01):
    df = pd.read_csv(file_path)
    
    # Nhóm sản phẩm theo Member_number
    transactions = df.groupby("Member_number")["itemDescription"].apply(list).tolist()
    
    # Mã hóa dữ liệu thành ma trận 0-1
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Áp dụng Apriori để tìm tập phổ biến
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    
    # Tìm tập phổ biến tối đại
    maximal_itemsets = frequent_itemsets[~frequent_itemsets['itemsets'].apply(lambda x: any(
        set(x).issubset(set(y)) for y in frequent_itemsets['itemsets'] if set(x) != set(y)))].copy()
    
    # Tìm tập phổ biến đóng
    closed_itemsets = frequent_itemsets[~frequent_itemsets['itemsets'].apply(lambda x: any(
        set(x).issubset(set(y)) and frequent_itemsets.loc[frequent_itemsets['itemsets'] == y, 'support'].values[0] == frequent_itemsets.loc[frequent_itemsets['itemsets'] == x, 'support'].values[0]
        for y in frequent_itemsets['itemsets'] if set(x) != set(y)))].copy()
    
    # Chuyển đổi frozenset thành danh sách dễ đọc
    frequent_itemsets["itemsets"] = frequent_itemsets["itemsets"].apply(lambda x: ', '.join(x))
    maximal_itemsets["itemsets"] = maximal_itemsets["itemsets"].apply(lambda x: ', '.join(x))
    closed_itemsets["itemsets"] = closed_itemsets["itemsets"].apply(lambda x: ', '.join(x))

    # Lưu file CSV
    frequent_itemsets.to_csv("frequent_itemsets.csv", index=False)
    maximal_itemsets.to_csv("maximal_itemsets.csv", index=False)
    closed_itemsets.to_csv("closed_itemsets.csv", index=False)

    return frequent_itemsets, maximal_itemsets, closed_itemsets

# Hàm hiển thị biểu đồ top 10 tập phổ biến
def plot_frequent_itemsets(frequent_itemsets):
    top_frequent = frequent_itemsets.nlargest(10, 'support')
    
    plt.figure(figsize=(12,6))
    plt.barh(top_frequent['itemsets'], top_frequent['support'], color='skyblue')
    plt.xlabel("Support")
    plt.ylabel("Itemsets")
    plt.title("Top 10 Frequent Itemsets")
    plt.gca().invert_yaxis()  # Đảo ngược trục Y để hiển thị đúng thứ tự
    plt.show()

# Hàm gợi ý sản phẩm dựa trên tập phổ biến
def suggest_products(product_name, itemsets_df):
    related_sets = itemsets_df[itemsets_df['itemsets'].str.contains(product_name, na=False)]
    
    recommendations = set()
    for items in related_sets['itemsets']:
        products = set(items.split(", "))
        products.discard(product_name)
        recommendations.update(products)

    return recommendations

# Chương trình chính
if __name__ == "__main__":
    file_path = select_csv_file()
    
    if file_path:
        print(f"Đang xử lý file: {file_path}")
        frequent_itemsets, maximal_itemsets, closed_itemsets = process_data(file_path)
        
        # Hiển thị dữ liệu
        print("\n📌 Top Frequent Itemsets:")
        print(frequent_itemsets.head())

        print("\n📌 Top Maximal Frequent Itemsets:")
        print(maximal_itemsets.head())

        print("\n📌 Top Closed Frequent Itemsets:")
        print(closed_itemsets.head())

        # Hiển thị biểu đồ
        plot_frequent_itemsets(frequent_itemsets)

        # Gợi ý sản phẩm
        product_name = input("\n🔍 Nhập sản phẩm để tìm gợi ý (hoặc bấm Enter để bỏ qua): ").strip()
        if product_name:
            recommendations = suggest_products(product_name, frequent_itemsets)
            print(f"\n💡 Sản phẩm gợi ý khi mua '{product_name}': {recommendations}")
    else:
        print("❌ Không có file nào được chọn.")
