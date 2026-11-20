import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import streamlit as st
import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))

def get_file_path(folder, filename):
    return os.path.join(base_dir, folder, filename)

st.header("โปรเจคการจำแนกข้อมูลดอกไม้")

col1, col2, col3 = st.columns(3)

# ลิงก์รูปภาพสำรองกรณีไม่มีไฟล์ในเครื่อง/GitHub
url_versicolor = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/320px-Iris_versicolor_3.jpg"
url_virginica = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/320px-Iris_virginica.jpg"
url_setosa = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/320px-Kosaciec_szczecinkowaty_Iris_setosa.jpg"

with col1:
    st.header("Versicolor")
    img1 = get_file_path("img", "iris1.jpg")
    st.image(img1 if os.path.exists(img1) else url_versicolor)

with col2:
    st.header("Virginica")
    img2 = get_file_path("img", "iris2.jpg")
    st.image(img2 if os.path.exists(img2) else url_virginica)

with col3:
    st.header("Setosa")
    img3 = get_file_path("img", "iris3.jpg")
    st.image(img3 if os.path.exists(img3) else url_setosa)

html_7 = """
<div style="background-color:#EC7063;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>สถิติข้อมูลดอกไม้</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")

# โหลดไฟล์ CSV (ถ้าหาไฟล์ในโฟลเดอร์ data ไม่เจอ จะดึงข้อมูล Iris มาตรฐานจาก scikit-learn มาใช้แทนอัตโนมัติ)
csv_path = get_file_path("data", "iris.csv")
if os.path.exists(csv_path):
    dt = pd.read_csv(csv_path)
else:
    iris = load_iris()
    dt = pd.DataFrame(iris.data, columns=['sepallength', 'sepalwidth', 'petallength', 'petalwidth'])
    # แปลงชื่อสายพันธุ์ให้เป็นตัวพิมพ์ใหญ่ตัวแรกตรงกับโค้ดเดิม
    dt['variety'] = [iris.target_names[i].capitalize() for i in iris.target]

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

pt_len = st.slider("กรุณาเลือกข้อมูล petal.length", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
pt_wd = st.slider("กรุณาเลือกข้อมูล petal.width", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

sp_len = st.number_input("กรุณาเลือกข้อมูล sepal.length", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
sp_wd = st.number_input("กรุณาเลือกข้อมูล sepal.width", min_value=0.0, max_value=10.0, value=3.5, step=0.1)

if st.button("ทำนายผล"):
    feature_cols = ['petallength', 'petalwidth', 'sepallength', 'sepalwidth']
    X = dt[feature_cols]
    y = dt['variety']
    
    Knn_model = KNeighborsClassifier(n_neighbors=3)
    Knn_model.fit(X, y)

    x_input = pd.DataFrame([[pt_len, pt_wd, sp_len, sp_wd]], columns=feature_cols)

    out = Knn_model.predict(x_input)
    st.success(f"ผลการทำนายคือ: **{out[0]}**")

    if out[0] == 'Versicolor':
        img_out = get_file_path("img", "iris1.jpg")
        st.image(img_out if os.path.exists(img_out) else url_versicolor)
    elif out[0] == 'Virginica':
        img_out = get_file_path("img", "iris2.jpg")
        st.image(img_out if os.path.exists(img_out) else url_virginica)
    else: # Setosa
        img_out = get_file_path("img", "iris3.jpg")
        st.image(img_out if os.path.exists(img_out) else url_setosa)
else:
    st.write("ไม่ทำนาย")