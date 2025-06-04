import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
import requests
import numpy as np

# --------------------
# 模型架構定義
# --------------------
class StatisticalAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        avg = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True)
        skew = ((x - avg) ** 3).mean(dim=[2, 3], keepdim=True) / (std + 1e-6) ** 3
        kurt = ((x - avg) ** 4).mean(dim=[2, 3], keepdim=True) / (std + 1e-6) ** 4
        stats = torch.cat([avg, std, skew, kurt], dim=1)
        weights = torch.softmax(stats.mean(dim=[2, 3]), dim=1).unsqueeze(-1).unsqueeze(-1)
        return x * weights.sum(dim=1, keepdim=True)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UNet_EdgeBranch_AttentionGate(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True))

        self.pool = nn.MaxPool2d(2)
        self.attn = StatisticalAttention()
        self.attn1 = StatisticalAttention()
        self.attn2 = StatisticalAttention()
        self.attn3 = StatisticalAttention()
        self.attn4 = StatisticalAttention()

        self.conv1 = conv_block(3, 32)
        self.conv2 = conv_block(32, 64)
        self.conv3 = conv_block(64, 128)
        self.conv4 = conv_block(128, 256)
        self.conv5 = conv_block(256, 512)

        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up6_conv = conv_block(256, 256)
        self.conv6 = conv_block(512, 256)

        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up7_conv = conv_block(128, 128)
        self.conv7 = conv_block(256, 128)

        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = conv_block(128, 64)

        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = conv_block(64, 32)

        self.final_mask = nn.Conv2d(32, 1, kernel_size=1)
        self.edge_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )

        self.ag6 = AttentionGate(256, 256, 128)
        self.ag7 = AttentionGate(128, 128, 64)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool(c1)
        p1 = self.attn1(p1)

        c2 = self.conv2(p1)
        p2 = self.pool(c2)
        p2 = self.attn2(p2)

        c3 = self.conv3(p2)
        p3 = self.pool(c3)
        p3 = self.attn3(p3)

        c4 = self.conv4(p3)
        p4 = self.pool(c4)
        p4 = self.attn4(p4)

        c5 = self.conv5(p4)
        c5 = self.attn(c5)

        u6 = self.up6(c5)
        u6 = self.up6_conv(u6)
        c4_attn = self.ag6(u6, c4)
        u6 = torch.cat([u6, c4_attn], dim=1)
        c6 = self.conv6(u6)

        u7 = self.up7(c6)
        u7 = self.up7_conv(u7)
        c3_attn = self.ag7(u7, c3)
        u7 = torch.cat([u7, c3_attn], dim=1)
        c7 = self.conv7(u7)

        u8 = self.up8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.conv8(u8)

        u9 = self.up9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.conv9(u9)

        mask_logits = self.final_mask(c9)
        edge_logits = self.edge_head(c9)
        return mask_logits, edge_logits

# --------------------
# 模型載入 + 預測
# --------------------
MODEL_URL = "https://huggingface.co/XiangtingLIN/unet-skin-lesion-model/resolve/main/unet_stat_attention_best.pth"
MODEL_PATH = "checkpoint/unet_stat_attention_best.pth"
IMG_SIZE = 256

@st.cache_resource
def load_model():
    os.makedirs("checkpoint", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        st.info("⏳ Downloading model from Hugging Face...")
        with open(MODEL_PATH, "wb") as f:
            f.write(requests.get(MODEL_URL).content)
        st.success("✅ Model downloaded!")
    model = UNet_EdgeBranch_AttentionGate()
    device = torch.device("cpu")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    return model, device

model, device = load_model()
transform_img = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

def predict(image: Image.Image):
    img_tensor = transform_img(image.resize((512, 512))).unsqueeze(0).to(device)
    with torch.no_grad():
        mask_logits, _ = model(img_tensor)
        pred_mask = (torch.sigmoid(mask_logits) > 0.5).float().cpu().numpy()[0, 0]
    pred_mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST)
    return image.resize((512, 512)), pred_mask_img

# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="Skin Lesion Segmentation", layout="centered")
st.markdown("""
    <style>
        .main { background-color: white; padding: 2rem; border-radius: 10px; }
        .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("功能選單")
option = st.sidebar.radio("選擇操作", ["上傳圖片", "模型預測"])

st.title("🧠 Skin Lesion Segmentation")
uploaded_file = st.file_uploader("請上傳一張圖片（jpg、png、bmp、jpeg）", type=["jpg", "png", "bmp", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="上傳的圖片", use_column_width=True)

    if st.button("開始預測"):
        with st.spinner("模型預測中..."):
            original, result = predict(image)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("原始圖片")
                st.image(original)
            with col2:
                st.subheader("預測遮罩")
                st.image(result)
        st.success("✅ 預測完成")
