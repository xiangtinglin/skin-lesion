import streamlit as st
from PIL import Image

# 設定頁面與白色背景
st.set_page_config(page_title="Skin Lesion Segmentation", layout="centered")

# 加上白色背景 CSS
st.markdown("""
    <style>
        .main {
            background-color: white;
            padding: 2rem;
            border-radius: 10px;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# 左側選單欄
st.sidebar.title("功能選單")
option = st.sidebar.radio("選擇操作", ["上傳圖片", "模型預測（尚未接）"])

# 主頁內容
st.title("上傳皮膚病灶圖片")
uploaded_file = st.file_uploader("請上傳一張圖片（格式：jpg、png、bmp...）", type=["jpg", "png", "bmp", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="上傳的圖片", use_column_width=True)
    
    if st.button("開始預測"):
        st.warning("⚠️ 尚未接上模型推論功能。")
