import streamlit as st
import os
from datetime import datetime
import pandas as pd
from experiment import run_experiments
import  re

st.set_page_config(page_title="水稻点云模型训练平台", layout="wide")
st.title("🌾 水稻点云模型训练平台")

st.markdown("### Step 1️⃣ 插秧日期与数据文件夹选择")
planting_date = st.date_input("📅 插秧日期", format="YYYY-MM-DD")
planting_date = datetime.combine(planting_date, datetime.min.time())
data_root = st.text_input("📁 输入数据文件夹路径", value="")

day_mapping = {}

if data_root and os.path.exists(data_root):
    folders = []
    for f in os.listdir(data_root):
        if os.path.isdir(os.path.join(data_root, f)):
            match = re.match(r"(\d{4})", f)  # 匹配前四位数字（MMDD）
            if match:
                folders.append(f)
    folders = sorted(folders)

    for folder in folders:
        try:
            mmdd = folder[:4]  # 提取前四位
            folder_date = datetime.strptime(f"2025{mmdd}", "%Y%m%d")
            delta_days = (folder_date - planting_date).days
            day_mapping[folder] = delta_days
        except ValueError:
            day_mapping[folder] = "解析失败"

    df_days = pd.DataFrame(list(day_mapping.items()), columns=["文件夹", "插秧后天数"])
    st.success("✅ 插秧后天数字典生成成功！")
    st.dataframe(df_days)
else:
    st.warning("⚠️ 请输入有效的数据路径，且该路径下包含如 '0806' 的四位文件夹名")

# Step 2：上传每个文件夹的 label 表格
st.markdown("### Step 2️⃣ 上传每个日期文件夹的标签表格（Excel 格式）")
uploaded_labels = {}
if folders:
    for folder in folders:
        uploaded_file = st.file_uploader(f"📄 上传 `{folder}` 文件夹对应的标签表格 (.xlsx)", type=["xlsx"], key=folder)
        if uploaded_file:
            uploaded_labels[folder] = uploaded_file

# Step 3：选择模型与配置
st.markdown("### Step 3️⃣ 选择模型与消融实验配置")
st.sidebar.header("模型增强模块")
model_options = ["dgcnn", "pointconv", "pointtransformer", "pct", "pointnet"]
selected_models = st.multiselect("🧠 选择要训练的模型", model_options, default=["dgcnn"])



use_attention = st.sidebar.checkbox("使用注意力机制", value=True)
use_residual = st.sidebar.checkbox("使用残差连接", value=True)
activation_choice = st.sidebar.selectbox("激活函数", ["relu", "leaky_relu", "gelu"])
use_feature_norm = st.sidebar.checkbox("使用特征正则化 (BatchNorm)", value=True)
use_rgb = st.checkbox("使用 RGB", value=True)
use_time = st.checkbox("使用时间（天数）", value=True)
epochs = st.number_input("训练轮数", min_value=1, value=100)
batch_size = st.number_input("Batch Size", min_value=1, value=16)
lr = st.number_input("学习率", value=0.001)





# Step 4：开始训练
if st.button("🚀 开始训练"):
    if not uploaded_labels or len(uploaded_labels) < len(folders):
        st.error("❌ 请为每个日期文件夹上传对应的标签文件")
    else:
        # 构造 configs 并运行
        configs = []
        for model in selected_models:
            configs.append({
                "model": model,
                "use_rgb": use_rgb,
                "use_time": use_time,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "use_attention": use_attention,
                "use_residual": use_residual,
                "activation": activation_choice,
                "use_feature_norm": use_feature_norm
            })

        st.info("开始训练中，请稍候...")

        run_experiments(
            configs=configs,
            data_root=data_root,
            label_files_dict=uploaded_labels,
            day_mapping=day_mapping  # 如果 `day_mapping` 不是必须参数，也可以删掉
        )
        st.success("🎉 所有模型训练完成！")
