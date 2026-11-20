import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# 1. ตั้งค่าหน้าเว็บ (ต้องอยู่บรรทัดแรกของ st)
st.set_page_config(page_title="โปรเจคตรวจโรคหัวใจ", page_icon="🫀", layout="wide")

# 2. เพิ่ม Custom CSS ปรับแต่งความสวยงาม
st.markdown("""
    <style>
    .title-text {
        font-size: 42px !important;
        font-weight: 700 !important;
        color: #E63946 !important;
        text-align: center;
        margin-bottom: 30px;
    }
    .header-box-red {
        background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 99%, #fecfef 100%);
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        color: #333;
        font-weight: bold;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .header-box-blue {
        background: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 100%);
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        color: #333;
        font-weight: bold;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- ส่วนหัวเรื่อง ---
st.markdown('<p class="title-text">🩺 โปรเจคการตรวจโรคหัวใจ</p>', unsafe_allow_html=True)

# --- ส่วนแสดงรูปภาพ (จัดให้อยู่ตรงกลางด้วยคอลัมน์) ---
img_col1, img_col2, img_col3, img_col4 = st.columns([1, 2, 2, 1]) 
with img_col2:
    st.markdown("<h4 style='text-align: center; color: #d32f2f;'>⚠️ เป็นโรคหัวใจ</h4>", unsafe_allow_html=True)
    st.image("./img/yes.jpg", use_container_width=True)
with img_col3:
    st.markdown("<h4 style='text-align: center; color: #388e3c;'>✅ ไม่เป็นโรคหัวใจ</h4>", unsafe_allow_html=True)
    st.image("./img/no.jpg", use_container_width=True)

st.divider()

# --- โหลดข้อมูล (เพิ่ม Cache เพื่อให้เว็บเร็วขึ้น) ---
@st.cache_data
def load_data():
    return pd.read_csv("./data/Heart3.csv")

dt = load_data()

# --- ส่วนข้อมูลสถิติ ---
st.markdown('<div class="header-box-red"><h5>📊 สถิติข้อมูลการเป็นโรคหัวใจ</h5></div>', unsafe_allow_html=True)

# ใช้ Expander ซ่อนตารางเพื่อให้หน้าเว็บดูสะอาดตา
with st.expander("คลิกเพื่อดูตารางข้อมูลและกราฟสถิติ 📉"):
    st.dataframe(dt.head(10), use_container_width=True)

    heart_counts = dt['HeartDisease'].value_counts().rename(
        index={0: "ไม่เป็นโรคหัวใจ (0)", 1: "เป็นโรคหัวใจ (1)"}
    )
    st.markdown("**กราฟแสดงสัดส่วนผู้ป่วย**")
    st.bar_chart(heart_counts, color="#ff4b4b")

st.divider()

# --- ส่วนฟอร์มรับค่าทำนาย ---
st.markdown('<div class="header-box-blue"><h5>🔍 กรอกข้อมูลเพื่อทำนายความเสี่ยง</h5></div>', unsafe_allow_html=True)

# จัดกลุ่ม Input ด้วย Columns ให้เป็นระเบียบ ไม่ยาวเกินไป
st.markdown("##### 👤 ข้อมูลทั่วไป")
c1, c2, c3 = st.columns(3)
with c1:
    age = st.slider("อายุ (Age)", min_value=20, max_value=100, value=50)
with c2:
    sex = st.selectbox("เพศ (Sex)", options=[1, 0], format_func=lambda x: "ชาย (1)" if x == 1 else "หญิง (0)")
with c3:
    chest_pain = st.selectbox(
        "ประเภทอาการเจ็บหน้าอก",
        options=[1, 2, 3, 4],
        format_func=lambda x: {1: "1-ไม่มีอาการ (ASY)", 2: "2-ไม่คงที่ (ATA)", 3: "3-ไม่ใช่โรคหัวใจ (NAP)", 4: "4-แบบทั่วไป (TA)"}[x]
    )

st.markdown("##### 🔬 ข้อมูลผลการตรวจร่างกาย")
c4, c5, c6 = st.columns(3)
with c4:
    resting_bp = st.number_input("ความดันโลหิตขณะพัก", min_value=0, max_value=250, value=130)
with c5:
    cholesterol = st.number_input("ระดับคอเลสเตอรอล", min_value=0, max_value=700, value=200)
with c6:
    fasting_bs = st.selectbox("น้ำตาลในเลือดหลังอดอาหาร > 120", options=[0, 1], format_func=lambda x: "ปกติ (0)" if x == 0 else "สูง (1)")

st.markdown("##### 💓 ข้อมูลคลื่นไฟฟ้าหัวใจและการออกกำลังกาย")
c7, c8, c9 = st.columns(3)
with c7:
    resting_ecg = st.selectbox(
        "คลื่นไฟฟ้าหัวใจขณะพัก",
        options=[1, 2, 3],
        format_func=lambda x: {1: "1-ห้องล่างซ้ายหน