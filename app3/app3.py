import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

# Hàm xử lý dữ liệu từ file CSV
def process_data(file, min_support=0.01):
    df = pd.read_csv(file.name)

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

    # Lưu hình ảnh và trả về
    plt.savefig("top_frequent_itemsets.png")
    return "top_frequent_itemsets.png"

# Hàm gợi ý sản phẩm dựa trên tập phổ biến
def suggest_products(product_name, itemsets_df):
    related_sets = itemsets_df[itemsets_df['itemsets'].str.contains(product_name, na=False)]

    recommendations = set()
    for items in related_sets['itemsets']:
        products = set(items.split(", "))
        products.discard(product_name)
        recommendations.update(products)

    return ", ".join(recommendations) if recommendations else "Không tìm thấy gợi ý nào."

# Hàm chính để tích hợp vào Gradio
def gradio_interface(file, min_support, product_name):
    if file is None:
        return "❌ Vui lòng tải lên file CSV!", None, None, None, None

    # Xử lý dữ liệu
    frequent_itemsets, maximal_itemsets, closed_itemsets = process_data(file, min_support)

    # Hiển thị biểu đồ
    plot_path = plot_frequent_itemsets(frequent_itemsets)

    # Gợi ý sản phẩm nếu có nhập
    recommendations = suggest_products(product_name, frequent_itemsets) if product_name else "Chưa nhập sản phẩm."

    return "✅ Xử lý xong!", frequent_itemsets, maximal_itemsets, closed_itemsets, plot_path, recommendations

# Tạo giao diện Gradio
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.File(label="📂 Chọn file CSV"),
        gr.Slider(minimum=0.01, maximum=1.0, value=0.05, label="⚙️ Min Support"),
        gr.Textbox(label="🔍 Nhập sản phẩm để tìm gợi ý (tùy chọn)")
    ],
    outputs=[
        "text",  # Trạng thái xử lý
        gr.Dataframe(label="📊 Frequent Itemsets"),
        gr.Dataframe(label="📊 Maximal Frequent Itemsets"),
        gr.Dataframe(label="📊 Closed Frequent Itemsets"),
        gr.Image(label="📈 Biểu đồ Top Frequent Itemsets"),
        "text"  # Gợi ý sản phẩm
    ],
    title="🛒 Phân Tích Giỏ Hàng - Apriori",
    description="Tải lên file CSV chứa dữ liệu giỏ hàng để tìm tập phổ biến, tối đại, đóng và gợi ý sản phẩm.",
    theme="huggingface"
)

# Chạy ứng dụng
iface.launch(share=True)
