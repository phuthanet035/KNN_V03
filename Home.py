import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================================
# 1. ตั้งค่าหน้าเว็บ (Page Configuration)
# ==========================================
st.set_page_config(
    page_title="ระบบประเมินความเสี่ยงโรคหัวใจ (KNN)",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. Custom CSS ตกแต่งความสวยงาม
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Prompt', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #FF4B4B 0%, #B71C1C 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.2);
    }
    .section-header {
        background-color: #F0F4F8;
        border-left: 6px solid #1E88E5;
        padding: 12px 20px;
        border-radius: 8px;
        margin: 20px 0 15px 0;
        color: #102A43;
    }
    .result-card-danger {
        background-color: #FFEBEE;
        border: 2px solid #EF5350;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        color: #B71C1C;
    }
    .result-card-safe {
        background-color: #E8F5E9;
        border: 2px solid #66BB6A;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        color: #1B5E20;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #1E88E5 0%, #1565C0 100%);
        color: white;
        font-size: 18px;
        font-weight: 600;
        border-radius: 10px;
        padding: 12px 24px;
        border: none;
        width: 100%;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(30, 136, 229, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. ฟังก์ชันโหลดข้อมูลและสร้างโมเดล (Caching)
# ==========================================
@st.cache_data
def load_data():
    file_path = "./data/Heart3.csv"
    if not os.path.exists(file_path):
        st.error(f"❌ ไม่พบไฟล์ข้อมูลที่ `{file_path}` กรุณาตรวจสอบโครงสร้างโฟลเดอร์บน GitHub")
        st.stop()
    return pd.read_csv(file_path)

@st.cache_resource
def train_knn_model(df, k_neighbors, use_scaler):
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    # แบ่งข้อมูลเพื่อวัดความแม่นยำเบื้องต้น (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if use_scaler:
        eval_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k_neighbors))
        final_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k_neighbors))
    else:
        eval_model = KNeighborsClassifier(n_neighbors=k_neighbors)
        final_model = KNeighborsClassifier(n_neighbors=k_neighbors)
        
    eval_model.fit(X_train, y_train)
    acc = accuracy_score(y_test, eval_model.predict(X_test))
    
    # เทรนด้วยข้อมูลทั้งหมดเพื่อใช้ทำนายจริง
    final_model.fit(X, y)
    return final_model, X.columns, acc

dt = load_data()

# ==========================================
# 4. แถบด้านข้าง (Sidebar) ตั้งค่าโมเดล
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/833/833472.png", width=80)
    st.title("⚙️ ตั้งค่าโมเดล KNN")
    st.markdown("ปรับแต่งพารามิเตอร์ของอัลกอริทึม K-Nearest Neighbors")
    
    k_val = st.slider("จำนวนเพื่อนบ้านใกล้เคียง (K)", min_value=1, max_value=15, value=5, step=2)
    use_scaler = st.checkbox("ปรับสเกลข้อมูล (StandardScaler)", value=True, help="แนะนำให้เปิดไว้ เพราะ KNN คำนวณจากระยะห่างของตัวเลข")
    
    model, feature_cols, model_acc = train_knn_model(dt, k_val, use_scaler)
    
    st.divider()
    st.metric(label="ความแม่นยำของโมเดล (Test Accuracy)", value=f"{model_acc*100:.2f}%")
    st.caption(f"จำนวนข้อมูลทั้งหมด: {len(dt)} รายการ")

# ==========================================
# 5. ส่วนหัวของเว็บ (Header)
# ==========================================
st.markdown("""
<div class="main-header">
    <h1>❤️ ระบบวิเคราะห์และทำนายความเสี่ยงโรคหัวใจ</h1>
    <p style="font-size:16px; margin:0; opacity:0.9;">จำแนกข้อมูลทางการแพทย์ด้วยเทคนิค K-Nearest Neighbors (KNN)</p>
</div>
""", unsafe_allow_html=True)

# แบ่ง Tab เพื่อความเป็นระเบียบ
tab1, tab2 = st.tabs(["🩺 แบบฟอร์มทำนายผล", "📊 สถิติและชุดข้อมูล (Dataset)"])

# ==========================================
# TAB 1: หน้ากรอกข้อมูลและทำนายผล
# ==========================================
with tab1:
    st.markdown('<div class="section-header"><h4>📝 กรอกข้อมูลสุขภาพของผู้เข้ารับการตรวจ</h4></div>', unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.subheader("ข้อมูลทั่วไปและอาการ")
            age = st.slider("1. อายุ (Age - ปี)", min_value=20, max_value=100, value=50)
            
            sex = st.selectbox(
                "2. เพศ (Sex)",
                options=[1, 0],
                format_func=lambda x: "ชาย (1)" if x == 1 else "หญิง (0)"
            )
            
            chest_pain = st.selectbox(
                "3. ประเภทอาการเจ็บหน้าอก (ChestPainType)",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "1 - ไม่มีอาการเจ็บหน้าอก (ASY)",
                    2: "2 - เจ็บหน้าอกแบบไม่คงที่ (ATA)",
                    3: "3 - เจ็บหน้าอกที่ไม่ใช่โรคหัวใจ (NAP)",
                    4: "4 - เจ็บหน้าอกแบบทั่วไป (TA)"
                }[x]
            )
            
            exercise_angina = st.selectbox(
                "4. เจ็บหน้าอกขณะออกกำลังกาย (ExerciseAngina)",
                options=[0, 1],
                format_func=lambda x: "ไม่เจ็บ (0)" if x == 0 else "เจ็บ (1)"
            )
            
            resting_bp = st.number_input("5. ความดันโลหิตขณะพัก (RestingBP - mm Hg)", min_value=0, max_value=250, value=130)
            cholesterol = st.number_input("6. ระดับคอเลสเตอรอล (Cholesterol - mg/dl)", min_value=0, max_value=700, value=200)

        with col2:
            st.subheader("ผลตรวจทางห้องปฏิบัติการและคลื่นหัวใจ")
            fasting_bs = st.selectbox(
                "7. น้ำตาลในเลือดหลังอดอาหาร > 120 mg/dl (FastingBS)",
                options=[0, 1],
                format_func=lambda x: "ปกติ <= 120 mg/dl (0)" if x == 0 else "สูง > 120 mg/dl (1)"
            )
            
            resting_ecg = st.selectbox(
                "8. คลื่นไฟฟ้าหัวใจขณะพัก (RestingECG)",
                options=[1, 2, 3],
                format_func=lambda x: {
                    1: "1 - กล้ามเนื้อหัวใจห้องล่างซ้ายหนา (LVH)",
                    2: "2 - ปกติ (Normal)",
                    3: "3 - คลื่น ST-T ผิดปกติ (ST)"
                }[x]
            )
            
            max_hr = st.slider("9. อัตราการเต้นของหัวใจสูงสุด (MaxHR - ครั้ง/นาที)", min_value=60, max_value=220, value=140)
            
            oldpeak = st.number_input("10. ค่า Oldpeak (ST depression)", min_value=-3.0, max_value=7.0, value=0.0, step=0.1)
            
            st_slope = st.selectbox(
                "11. ความชันของคลื่น ST (ST_Slope)",
                options=[1, 2, 3],
                format_func=lambda x: {
                    1: "1 - ลาดลง (Down)",
                    2: "2 - ราบเรียบ (Flat)",
                    3: "3 - ลาดขึ้น (Up)"
                }[x]
            )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 วิเคราะห์และทำนายผล")

    # แสดงผลลัพธ์เมื่อกดปุ่ม
    if submitted:
        x_input = pd.DataFrame(
            [[age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
              resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]],
            columns=feature_cols
        )
        
        prediction = model.predict(x_input)[0]
        proba = model.predict_proba(x_input)[0]
        
        st.markdown('<div class="section-header"><h4>📋 ผลการวิเคราะห์ความเสี่ยง</h4></div>', unsafe_allow_html=True)
        
        res_col1, res_col2 = st.columns([1, 2])
        
        if prediction == 1:
            with res_col1:
                if os.path.exists("./img/yes.jpg"):
                    st.image("./img/yes.jpg", use_container_width=True)
                else:
                    st.markdown("<h1 style='text-align:center; font-size:80px;'>⚠️</h1>", unsafe_allow_html=True)
            with res_col2:
                st.markdown(f"""
                <div class="result-card-danger">
                    <h2>⚠️ ผลการทำนาย: มีความเสี่ยงเป็นโรคหัวใจ (Class 1)</h2>
                    <p style="font-size:18px;">ความน่าจะเป็นที่ประเมินได้จากเพื่อนบ้าน {k_val} ลำดับแรก: <b>{proba[1]*100:.1f}%</b></p>
                    <p style="font-size:14px; color:#555;">*หมายเหตุ: ผลลัพธ์นี้เป็นการคำนวณทางสถิติเบื้องต้น ควรปรึกษาแพทย์ผู้เชี่ยวชาญเพื่อตรวจวินิจฉัยอย่างละเอียด</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            with res_col1:
                if os.path.exists("./img/no.jpg"):
                    st.image("./img/no.jpg", use_container_width=True)
                else:
                    st.markdown("<h1 style='text-align:center; font-size:80px;'>💚</h1>", unsafe_allow_html=True)
            with res_col2:
                st.markdown(f"""
                <div class="result-card-safe">
                    <h2>✅ ผลการทำนาย: ไม่พบความเสี่ยงโรคหัวใจ (Class 0)</h2>
                    <p style="font-size:18px;">ความน่าจะเป็นที่สุขภาพปกติจากเพื่อนบ้าน {k_val} ลำดับแรก: <b>{proba[0]*100:.1f}%</b></p>
                    <p style="font-size:14px; color:#555;">*แนะนำให้รักษาสุขภาพ ออกกำลังกายสม่ำเสมอ และตรวจสุขภาพประจำปี</p>
                </div>
                """, unsafe_allow_html=True)

# ==========================================
# TAB 2: หน้าแสดงสถิติและข้อมูลตัวอย่าง
# ==========================================
with tab2:
    st.markdown('<div class="section-header"><h4>📈 ภาพรวมชุดข้อมูล (Heart3.csv)</h4></div>', unsafe_allow_html=True)
    
    m1, m2, m3 = st.columns(3)
    total_cases = len(dt)
    disease_cases = int(dt['HeartDisease'].sum())
    normal_cases = total_cases - disease_cases
    
    m1.metric("จำนวนผู้ป่วยทั้งหมด", f"{total_cases} คน")
    m2.metric("กลุ่มเสี่ยง/เป็นโรคหัวใจ (1)", f"{disease_cases} คน", f"{(disease_cases/total_cases)*100:.1f}%", delta_color="inverse")
    m3.metric("กลุ่มปกติ (0)", f"{normal_cases} คน", f"{(normal_cases/total_cases)*100:.1f}%")
    
    st.subheader("สัดส่วนผู้ที่เป็นและไม่เป็นโรคหัวใจ")
    heart_counts = dt['HeartDisease'].value_counts().rename(
        index={0: "ไม่เป็นโรคหัวใจ (0)", 1: "เป็นโรคหัวใจ (1)"}
    )
    st.bar_chart(heart_counts, color="#FF4B4B")
    
    st.subheader("ตัวอย่างข้อมูล 10 แถวแรก")
    st.dataframe(dt.head(10), use_container_width=True)