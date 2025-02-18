import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

# Xử lý dữ liệu CSV
def process_data(file, min_support=0.01):
    df = pd.read_csv(file.name)
    transactions = df.groupby("Member_number")["itemDescription"].apply(list).tolist()

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

    maximal_itemsets = frequent_itemsets[~frequent_itemsets['itemsets'].apply(lambda x: any(
        set(x).issubset(set(y)) for y in frequent_itemsets['itemsets'] if set(x) != set(y)))].copy()

    closed_itemsets = frequent_itemsets[~frequent_itemsets['itemsets'].apply(lambda x: any(
        set(x).issubset(set(y)) and frequent_itemsets.loc[frequent_itemsets['itemsets'] == y, 'support'].values[0] == frequent_itemsets.loc[frequent_itemsets['itemsets'] == x, 'support'].values[0]
        for y in frequent_itemsets['itemsets'] if set(x) != set(y)))].copy()

    frequent_itemsets["itemsets"] = frequent_itemsets["itemsets"].apply(lambda x: ', '.join(x))
    maximal_itemsets["itemsets"] = maximal_itemsets["itemsets"].apply(lambda x: ', '.join(x))
    closed_itemsets["itemsets"] = closed_itemsets["itemsets"].apply(lambda x: ', '.join(x))

    return frequent_itemsets, maximal_itemsets, closed_itemsets

# Vẽ biểu đồ top tập phổ biến
def plot_frequent_itemsets(frequent_itemsets):
    top_frequent = frequent_itemsets.nlargest(10, 'support')

    plt.figure(figsize=(10, 5))
    plt.barh(top_frequent['itemsets'], top_frequent['support'], color='#007bff')
    plt.xlabel("Support")
    plt.ylabel("Itemsets")
    plt.title("🔥 Top 10 Frequent Itemsets")
    plt.gca().invert_yaxis()

    plt.savefig("top_frequent_itemsets.png")
    return "top_frequent_itemsets.png"

# Gợi ý sản phẩm dựa trên tập phổ biến
def suggest_products(product_name, itemsets_df):
    related_sets = itemsets_df[itemsets_df['itemsets'].str.contains(product_name, na=False)]
    recommendations = set()
    
    for items in related_sets['itemsets']:
        products = set(items.split(", "))
        products.discard(product_name)
        recommendations.update(products)

    return ", ".join(recommendations) if recommendations else "❌ Không tìm thấy gợi ý."

# Giao diện Gradio
def gradio_interface(file, min_support, product_name):
    if file is None:
        return "❌ Vui lòng tải file CSV!", None, None, None, None, None

    frequent_itemsets, maximal_itemsets, closed_itemsets = process_data(file, min_support)
    plot_path = plot_frequent_itemsets(frequent_itemsets)
    recommendations = suggest_products(product_name, frequent_itemsets) if product_name else "🔍 Chưa nhập sản phẩm."

    return "✅ Hoàn thành!", frequent_itemsets, maximal_itemsets, closed_itemsets, plot_path, recommendations

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
    gr.Markdown("### 📂 Tải lên file CSV để tìm tập phổ biến, tập tối đại, tập đóng và gợi ý sản phẩm")

    with gr.Row():
        file_input = gr.File(label="📂 Chọn file CSV")
        min_support_input = gr.Slider(minimum=0.01, maximum=1.0, value=0.05, label="⚙️ Min Support")
    
    product_input = gr.Textbox(label="🔍 Nhập sản phẩm để tìm gợi ý (tùy chọn)")

    run_button = gr.Button("🚀 Phân tích ngay", variant="primary")

    with gr.Row():
        status_output = gr.Textbox(label="📢 Trạng thái", interactive=False)

    with gr.Tab("📊 Frequent Itemsets"):
        frequent_itemsets_output = gr.Dataframe()
    
    with gr.Tab("📊 Maximal Frequent Itemsets"):
        maximal_itemsets_output = gr.Dataframe()
    
    with gr.Tab("📊 Closed Frequent Itemsets"):
        closed_itemsets_output = gr.Dataframe()
    
    with gr.Row():
        chart_output = gr.Image(label="📈 Biểu đồ Top Frequent Itemsets")
    
    with gr.Row():
        suggestion_output = gr.Textbox(label="🎯 Gợi ý sản phẩm", interactive=False)

    run_button.click(
        gradio_interface, 
        inputs=[file_input, min_support_input, product_input], 
        outputs=[status_output, frequent_itemsets_output, maximal_itemsets_output, closed_itemsets_output, chart_output, suggestion_output]
    )

# Chạy ứng dụng
demo.launch(share=True)
