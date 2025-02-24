import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

# Xử lý dữ liệu CSV - nhóm các sản phẩm theo từng khách hàng
# Xử lý dữ liệu CSV - nhóm các sản phẩm theo từng khách hàng
def process_data(file, min_support=0.01):
    # đọc dữ liệu từ file csv
    df = pd.read_csv(file.name)
    transactions = df.groupby("Member_number")["itemDescription"].apply(list).tolist()

    # In ra một số giao dịch đầu tiên để kiểm tra dữ liệu
    print("🔍 Một số giao dịch mẫu:")
    for i, transaction in enumerate(transactions[:5]):  # Hiển thị 5 giao dịch đầu tiên
        print(f"Khách hàng {i + 1}: {transaction}")

    # chuyển đổi dữ liệu - mã hóa giỏ hàng thành dạng One-Hot Encoding
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # dùng thuật toán Apiori để tìm các tập phổ biến từ dữ liệu
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

    # In một số tập phổ biến ban đầu để kiểm tra
    print("\n📊 Một số tập phổ biến đầu tiên:")
    print(frequent_itemsets.head(5))

    # tìm tập tối đại - loại bỏ các tập con của các tập phổ biến khác
    maximal_itemsets = frequent_itemsets[~frequent_itemsets['itemsets'].apply(lambda x: any(
        set(x).issubset(set(y)) for y in frequent_itemsets['itemsets'] if set(x) != set(y)))].copy()

    # tìm tập đóng - loại bỏ tập phổ biến nếu tồn tại một tập chứa nó có cùng giá trị
    closed_itemsets = frequent_itemsets[~frequent_itemsets['itemsets'].apply(lambda x: any(
        set(x).issubset(set(y)) and frequent_itemsets.loc[frequent_itemsets['itemsets'] == y, 'support'].values[0] == frequent_itemsets.loc[frequent_itemsets['itemsets'] == x, 'support'].values[0]
        for y in frequent_itemsets['itemsets'] if set(x) != set(y)))].copy()

    # In một số tập tối đại và tập đóng
    print("\n📌 Một số tập tối đại đầu tiên:")
    print(maximal_itemsets.head(5))

    print("\n🔒 Một số tập đóng đầu tiên:")
    print(closed_itemsets.head(5))

    return frequent_itemsets, maximal_itemsets, closed_itemsets, len(frequent_itemsets), len(maximal_itemsets), len(closed_itemsets)

# Vẽ biểu đồ top 10 tập phổ biến nhất dựa trên giá trị là support
def plot_frequent_itemsets(frequent_itemsets):
    # chọn 10 tập phổ biến có giá trị support cao nhất
    top_frequent = frequent_itemsets.nlargest(10, 'support')

    # vẽ biểu đồ bar chart
    plt.figure(figsize=(10, 5))
    plt.barh(top_frequent['itemsets'], top_frequent['support'], color='#007bff')
    plt.xlabel("Support")
    plt.ylabel("Itemsets")
    plt.title("🔥 Top 10 Frequent Itemsets")
    plt.gca().invert_yaxis()

    # lưu biểu đồ thành ảnh png
    plt.savefig("top_frequent_itemsets.png")
    return "top_frequent_itemsets.png"

# Gợi ý sản phẩm dựa trên tập phổ biến chứa tên sản phẩm đó (product_name)
def suggest_products(product_name, itemsets_df):
    # tìm tất cả tập phổ biến chứa sản phẩm cần tìm
    related_sets = itemsets_df[itemsets_df['itemsets'].str.contains(product_name, na=False)]
    recommendations = set()
    
    # loại bỏ sản phẩm chính khỏi tập hợp
    for items in related_sets['itemsets']:
        products = set(items.split(", "))
        products.discard(product_name)
        recommendations.update(products)

    # trả về danh sách sản phẩm gợi ý
    return list(recommendations) if recommendations else ["❌🙄 Không tìm thấy gợi ý."]

# Giao diện Gradio
def gradio_interface(file, min_support, product_name):
    if file is None:
        return "❌🙄 Vui lòng tải file CSV!", None, None, None, None, None

    # xử lý dữ liệu từ file csv truyền vào
    frequent_itemsets, maximal_itemsets, closed_itemsets, frequent_count, maximal_count, closed_count = process_data(file, min_support)
    
    # vẽ biểu đồ với hàm vẽ biểu đồ plot_frequent_itemsets
    plot_path = plot_frequent_itemsets(frequent_itemsets)

    # tìm gợi ý sản phẩm với hàm suggest_products
    recommendations = suggest_products(product_name, frequent_itemsets) if product_name else "❌🙄 Chưa nhập sản phẩm."

    return "✅😊 Hoàn thành!", frequent_itemsets, maximal_itemsets, closed_itemsets, plot_path, recommendations, frequent_count, maximal_count, closed_count


# Tuỳ chỉnh CSS
custom_css = """
h1, h2 {
    text-align: center;
}
.gr-file {
    border: 2px dashed #007bff !important;
}
.gr-textbox {
    border-radius: 10px;
}
"""

# Giao diện Blocks() của Gradio
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 🛒 Phân Tích Giỏ Hàng - Apriori")
    gr.Markdown("### 📂😎 Tải lên file CSV để tìm tập phổ biến, tập tối đại, tập đóng và gợi ý sản phẩm")

    with gr.Row():
        # interface lấy input là file
        file_input = gr.File(label="📂😊 Chọn file CSV")
        
        # interface lấy input là giá trị số, có hỗ trợ slider
        min_support_input = gr.Slider(minimum=0.01, maximum=1.0, value=0.05, label="⚙️ Min Support")
    
    # interface lấy input là text
    product_input = gr.Textbox(label="🔍😍 Nhập sản phẩm để tìm gợi ý (tùy chọn)")

    # interface button
    run_button = gr.Button("🚀 Phân tích ngay", variant="primary")

    # tạo 1 hàng hiển thị output của gợi ý sản phẩm
    with gr.Row():
        suggestion_output = gr.Dataframe(label="🎯😍 Gợi ý sản phẩm", interactive=False, headers=["🔹 Sản phẩm gợi ý"])

    # tạo 1 hàng hiển thị output trạng thái của chương trình
    with gr.Row():
        status_output = gr.Textbox(label="📢😉 Trạng thái", interactive=False)

    # tạo các interface table nằm trên các tabs, mỗi tab là một tập
    with gr.Tab("🗃️ Frequent Itemsets"):
        frequent_itemsets_output = gr.Dataframe()
    
    with gr.Tab("🗃️ Maximal Frequent Itemsets"):
        maximal_itemsets_output = gr.Dataframe()
    
    with gr.Tab("🗃️ Closed Frequent Itemsets"):
        closed_itemsets_output = gr.Dataframe()
    
    # tạo 1 row hiển thị số lượng của từng tập
    with gr.Row():
        frequent_count_output = gr.Textbox(label="📊 Số lượng tập phổ biến", interactive=False)
        maximal_count_output = gr.Textbox(label="📊 Số lượng tập tối đại", interactive=False)
        closed_count_output = gr.Textbox(label="📊 Số lượng tập đóng", interactive=False)


    # tạo 1 row mới chứa interface output của vẽ biểu đồ 10 tập có suggest cao nhất (best seller)
    with gr.Row():
        chart_output = gr.Image(label="📈 Biểu đồ Top Frequent Itemsets")
    
    
    # xử lý sự kiện click của button nhận input truyền vào gradio_interface xử lý và trả về giá trị và gán lên giao diện
    run_button.click(
        gradio_interface, 
        inputs=[file_input, min_support_input, product_input], 
        outputs=[
        status_output, frequent_itemsets_output, maximal_itemsets_output, closed_itemsets_output, 
        chart_output, suggestion_output, frequent_count_output, maximal_count_output, closed_count_output
        ]
    )

# Chạy ứng dụng
demo.launch(share=True)
