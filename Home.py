import os
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np

# หาตำแหน่งโฟลเดอร์ปัจจุบันของไฟล์ Home.py เพื่อให้ดึงรูปภาพและไฟล์ CSV ได้ไม่พลาด
base_dir = os.path.dirname(os.path.abspath(__file__))

def get_file_path(folder, filename):
    return os.path.join(base_dir, folder, filename)

st.header("โปรเจคการจำแนกข้อมูลดอกไม้")

# แก้ไข: เพิ่ม col3 เข้าไปใน st.columns(3) จากเดิมที่มีแค่ col1, col2
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Versicolor")
    img1 = get_file_path("img", "iris1.jpg")
    if os.path.exists(img1):
        st.image(img1)
    else:
        st.warning("ไม่พบไฟล์ img/iris1.jpg")

with col2:
    st.header("Virginica")
    img2 = get_file_path("img", "iris2.jpg")
    if os.path.exists(img2):
        st.image(img2)
    else:
        st.warning("ไม่พบไฟล์ img/iris2.jpg")

with col3:
    st.header("Setosa")
    img3 = get_file_path("img", "iris3.jpg")
    if os.path.exists(img3):
        st.image(img3)
    else:
        st.warning("ไม่พบไฟล์ img/iris3.jpg")

html_7 = """
<div style="background-color:#EC7063;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>สถิติข้อมูลดอกไม้</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")

# โหลดไฟล์ CSV
csv_path = get_file_path("data", "iris.csv")
dt = pd.read_csv(csv_path)
st.write(dt.head(10))

dt1 = dt['petallength'].sum()
dt2 = dt['petalwidth'].sum()
dt3 = dt['sepallength'].sum()
dt4 = dt['sepalwidth'].sum()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["petallength", "petalwidth", "sepallength", "sepalwidth"], columns=["Total"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
    st.bar_chart(dx2)
else:
    st.write("ไม่แสดงข้อมูล")

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

# ปรับค่า min_value, max_value, step ให้เหมาะสมกับขนาดดอกไม้จริง (หน่วยเป็น cm)
pt_len = st.slider("กรุณาเลือกข้อมูล petal.length", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
pt_wd = st.slider("กรุณาเลือกข้อมูล petal.width", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

sp_len = st.number_input("กรุณาเลือกข้อมูล sepal.length", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
sp_wd = st.number_input("กรุณาเลือกข้อมูล sepal.width", min_value=0.0, max_value=10.0, value=3.5, step=0.1)

if st.button("ทำนายผล"):
    # ระบุลำดับคอลัมน์ให้ตรงกับข้อมูลที่รับเข้ามา (x_input)
    feature_cols = ['petallength', 'petalwidth', 'sepallength', 'sepalwidth']
    X = dt[feature_cols]
    y = dt['variety']
    
    Knn_model = KNeighborsClassifier(n_neighbors=3)
    Knn_model.fit(X, y)

    # สร้าง DataFrame ที่มีชื่อคอลัมน์ตรงกัน เพื่อป้องกัน Warning และทำนายได้แม่นยำ
    x_input = pd.DataFrame([[pt_len, pt_wd, sp_len, sp_wd]], columns=feature_cols)

    out = Knn_model.predict(x_input)
    st.success(f"ผลการทำนายคือ: **{out[0]}**")

    # แสดงรูปภาพตามสายพันธุ์ที่ทำนายได้ (จัดให้ตรงกับหัวข้อด้านบน)
    if out[0] == 'Versicolor':
        img_out = get_file_path("img", "iris1.jpg")
    elif out[0] == 'Virginica':
        img_out = get_file_path("img", "iris2.jpg")
    else: # Setosa
        img_out = get_file_path("img", "iris3.jpg")
        
    if os.path.exists(img_out):
        st.image(img_out)
else:
    st.write("ไม่ทำนาย")