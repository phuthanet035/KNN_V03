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
    page_title="CARDIO-AI | ระบบวิเคราะห์โรคหัวใจอัจฉริยะ",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. Custom CSS สไตล์ Cyber-Medical Glassmorphism
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&family=Orbitron:wght@500;700&display=swap');

    /* พื้นหลังหลักโทน Midnight Navy สบายตา */
    .stApp {
        background: radial-gradient(circle at 15% 20%, #162238 0%, #0B1120 50%, #060913 100%);
        color: #E2E8F0;
        font-family: 'Prompt', sans-serif;
    }

    /* ปรับแต่ง Sidebar ให้เป็นกระจกเข้ม */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(56, 189, 248, 0.2);
    }

    /* ฟอนต์ตัวเลขและหัวข้อภาษาอังกฤษสไตล์ Sci-Fi */
    .cyber-font {
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 1.5px;
    }

    /* กล่อง Hero Header พร้อมแสง Glow */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(56, 189, 248, 0.35);
        box-shadow: 0 0 35px rgba(14, 165, 233, 0.15), inset 0 0 20px rgba(56, 189, 248, 0.05);
        border-radius: 24px;
        padding: 35px 25px;
        text-align: center;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38BDF8, #818CF8, #F472B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    /* แอนิเมชันเส้นคลื่นหัวใจ ECG */
    .ecg-line {
        stroke-dasharray: 1000;
        stroke-dashoffset: 1000;
        animation: dash 3.5s linear infinite;
    }
    @keyframes dash {
        to { stroke-dashoffset: 0; }
    }

    /* กล่อง Glassmorphism สำหรับแบ่งโซน */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
    }

    .section-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.4);
        padding: 5px 14px;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 12px;
    }

    /* การ์ดแสดงผลลัพธ์ - เสี่ยงสูง (Coral Neon สบายตา) */
    .result-danger {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.45) 0%, rgba(30, 41, 59, 0.85) 100%);
        backdrop-filter: blur(16px);
        border: 2px solid #FB7185;
        box-shadow: 0 0 30px rgba(251, 113, 133, 0.25);
        border-radius: 22px;
        padding: 30px;
        text-align: center;
    }

    /* การ์ดแสดงผลลัพธ์ - ปลอดภัย (Emerald Cyan สบายตา) */
    .result-safe {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.45) 0%, rgba(30, 41, 59, 0.85) 100%);
        backdrop-filter: blur(16px);
        border: 2px solid #34D399;
        box-shadow: 0 0 30px rgba(52, 211, 153, 0.25);
        border-radius: 22px;
        padding: 30px;
        text-align: center;
    }

    /* ปุ่มกดเรืองแสงสุดเท่ */
    div.stFormSubmitButton > button {
        background: linear-gradient(90deg, #0284C7 0%, #4F46E5 50%, #7C3AED 100%);
        background-size: 200% auto;
        color: white !important;
        font-family: 'Prompt', sans-serif;
        font-size: 19px !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 15px 30px !important;
        width: 100%;
        box-shadow: 0 0 25px rgba(79, 70, 229, 0.4);
        transition: 0.4s ease !important;
    }
    div.stFormSubmitButton > button:hover {
        background-position: right center;
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 0 35px rgba(56, 189, 248, 0.65);
    }

    /* ปรับแต่ง Tab */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 8px;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.15);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #94A3B8;
        font-weight: 500;
        padding: 10px 24px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, rgba(56, 189, 248, 0.2), rgba(129, 140, 248, 0.2)) !important;
        color: #38BDF8 !important;
        border: 1px solid rgba(56, 189, 248, 0.4);
    }

    /* ปรับสี Label ของ Input ให้อ่านง่ายสบายตา */
    label, .stMarkdown p {
        color: #CBD5E1 !important;
    }
    h1, h2, h3, h4 {
        color: #F8FAFC !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. โหลดข้อมูลและสร้างโมเดล KNN
# ==========================================
@st.cache_data
def load_data():
    file_path = "./data/Heart3.csv"
    if not os.path.exists(file_path):
        st.error(f"❌ ไม่พบไฟล์ข้อมูลที่ `{file_path}` กรุณาตรวจสอบโฟลเดอร์ data/ บน GitHub")
        st.stop()
    return pd.read_csv(file_path)

@st.cache_resource
def train_knn_model(df, k_neighbors, use_scaler):
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if use_scaler:
        eval_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k_neighbors))
        final_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k_neighbors))
    else:
        eval_model = KNeighborsClassifier(n_neighbors=k_neighbors)
        final_model = KNeighborsClassifier(n_neighbors=k_neighbors)
        
    eval_model.fit(X_train, y_train)
    acc = accuracy_score(y_test, eval_model.predict(X_test))
    
    final_model.fit(X, y)
    return final_model, X.columns, acc

dt = load_data()

# ==========================================
# 4. แถบควบคุมด้านข้าง (Cyber Control Sidebar)
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0;">
        <span style="font-size: 50px;">🫀</span>
        <h2 class="cyber-font" style="color:#38BDF8; margin:5px 0;">CARDIO-AI</h2>
        <p style="font-size:12px; color:#94A3B8;">KNN DIAGNOSTIC ENGINE v2.0</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### 🎛️ ปรับแต่งสมองกล (KNN)")
    
    k_val = st.slider(
        "ค่า K (จำนวนเพื่อนบ้านใกล้เคียง)",
        min_value=1, max_value=15, value=5, step=2,
        help="กำหนดจำนวนข้อมูลที่ใกล้เคียงที่สุดเพื่อใช้โหวตผลลัพธ์"
    )
    use_scaler = st.toggle(
        "เปิดใช้ StandardScaler",
        value=True,
        help="ปรับสเกลข้อมูลให้เท่าเทียมกันก่อนคำนวณระยะห่าง (แนะนำ)"
    )
    
    model, feature_cols, model_acc = train_knn_model(dt, k_val, use_scaler)
    
    st.markdown(f"""
    <div style="background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 14px; padding: 15px; margin-top: 15px; text-align: center;">
        <div style="font-size: 12px; color: #94A3B8;">ความแม่นยำของโมเดล (ACCURACY)</div>
        <div class="cyber-font" style="font-size: 28px; font-weight: 700; color: #34D399;">{model_acc*100:.2f}%</div>
        <div style="font-size: 11px; color: #64748B; margin-top:4px;">ฐานข้อมูลผู้ป่วย {len(dt):,} รายการ</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 5. ส่วนหัวสุดอลังการ (Hero Header + ECG Wave)
# ==========================================
st.markdown("""
<div class="hero-container">
    <div class="section-badge cyber-font">AI-POWERED CARDIAC SCREENING</div>
    <div class="hero-title">ระบบวิเคราะห์และทำนายความเสี่ยงโรคหัวใจ</div>
    <p style="color:#94A3B8; font-size:16px; max-width:650px; margin:0 auto 15px auto;">
        ประมวลผลข้อมูลสุขภาพเชิงลึก 11 มิติ ด้วยอัลกอริทึม <b>K-Nearest Neighbors (KNN)</b> เพื่อประเมินความเสี่ยงอย่างแม่นยำและรวดเร็ว
    </p>
    <svg width="280" height="45" viewBox="0 0 280 45" style="margin: 0 auto; display: block;">
        <path class="ecg-line" fill="none" stroke="#38BDF8" stroke-width="2.5"
              d="M0,25 L60,25 L75,10 L85,38 L98,3 L112,42 L122,25 L160,25 L175,15 L185,33 L195,25 L280,25" />
    </svg>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 6. เมนูแท็บหลัก (Tabs)
# ==========================================
tab1, tab2 = st.tabs(["⚡ วิเคราะห์และทำนายผล (Prediction)", "📊 ฐานข้อมูลและสถิติ (Analytics)"])

# ------------------------------------------
# TAB 1: หน้าป้อนข้อมูลและทำนายผล
# ------------------------------------------
with tab1:
    with st.form("prediction_form"):
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
            <div class="glass-card">
                <span class="section-badge">STEP 01</span>
                <h4 style="margin-top:0;">🧬 ข้อมูลพื้นฐานและอาการทางกายภาพ</h4>
                <p style="font-size:13px; color:#94A3B8;">ระบุข้อมูลส่วนตัวและลักษณะอาการเจ็บหน้าอกที่พบ</p>
            </div>
            """, unsafe_allow_html=True)
            
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
            st.markdown("""
            <div class="glass-card">
                <span class="section-badge">STEP 02</span>
                <h4 style="margin-top:0;">📈 ค่าผลตรวจทางคลินิกและคลื่นหัวใจ</h4>
                <p style="font-size:13px; color:#94A3B8;">ระบุค่าผลตรวจเลือด อัตราการเต้นหัวใจ และคลื่นไฟฟ้าหัวใจ</p>
            </div>
            """, unsafe_allow_html=True)
            
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
        submitted = st.form_submit_button("🚀 เริ่มการวิเคราะห์ความเสี่ยงด้วย AI")

    # ==========================================
    # แสดงผลลัพธ์การทำนาย (Result Display)
    # ==========================================
    if submitted:
        x_input = pd.DataFrame(
            [[age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
              resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]],
            columns=feature_cols
        )
        
        prediction = model.predict(x_input)[0]
        proba = model.predict_proba(x_input)[0]
        risk_percent = proba[1] * 100
        safe_percent = proba[0] * 100
        
        st.markdown("<br>", unsafe_allow_html=True)
        res_col1, res_col2 = st.columns([1, 2.2], gap="large")
        
        if prediction == 1:
            with res_col1:
                if os.path.exists("./img/yes.jpg"):
                    st.image("./img/yes.jpg", use_container_width=True)
                else:
                    st.markdown("""
                    <div class="result-danger" style="padding: 40px 20px;">
                        <div style="font-size: 75px;">🚨</div>
                        <div class="cyber-font" style="color:#FB7185; font-weight:700; margin-top:10px;">HIGH RISK</div>
                    </div>
                    """, unsafe_allow_html=True)
            with res_col2:
                st.markdown(f"""
                <div class="result-danger">
                    <span style="background:rgba(251,113,133,0.2); color:#FB7185; padding:4px 12px; border-radius:20px; font-size:13px; font-weight:600;">
                        ผลการจำแนกข้อมูล: CLASS 1
                    </span>
                    <h2 style="color:#FDA4AF !important; margin: 12px 0;">⚠️ ตรวจพบความเสี่ยงในการเป็นโรคหัวใจ</h2>
                    <div style="font-size: 18px; color:#E2E8F0; margin-bottom: 15px;">
                        ดัชนีความเสี่ยงที่คำนวณได้: <span class="cyber-font" style="font-size:28px; color:#FB7185; font-weight:700;">{risk_percent:.1f}%</span>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.7); border-radius: 10px; height: 14px; width: 100%; overflow: hidden; margin-bottom: 15px;">
                        <div style="background: linear-gradient(90deg, #F43F5E, #FB7185); width: {risk_percent}%; height: 100%;"></div>
                    </div>
                    <p style="font-size: 14px; color: #CBD5E1; margin:0;">
                        คำแนะนำ: ข้อมูลของคุณมีลักษณะใกล้เคียงกับกลุ่มผู้ป่วยโรคหัวใจในฐานข้อมูล (K={k_val}) ควรเข้าพบแพทย์เฉพาะทางด้านโรคหัวใจเพื่อตรวจคลื่นไฟฟ้าหัวใจและวินิจฉัยอย่างละเอียด
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            with res_col1:
                if os.path.exists("./img/no.jpg"):
                    st.image("./img/no.jpg", use_container_width=True)
                else:
                    st.markdown("""
                    <div class="result-safe" style="padding: 40px 20px;">
                        <div style="font-size: 75px;">🛡️</div>
                        <div class="cyber-font" style="color:#34D399; font-weight:700; margin-top:10px;">LOW RISK</div>
                    </div>
                    """, unsafe_allow_html=True)
            with res_col2:
                st.markdown(f"""
                <div class="result-safe">
                    <span style="background:rgba(52,211,153,0.2); color:#34D399; padding:4px 12px; border-radius:20px; font-size:13px; font-weight:600;">
                        ผลการจำแนกข้อมูล: CLASS 0
                    </span>
                    <h2 style="color:#6EE7B7 !important; margin: 12px 0;">✅ ไม่พบความเสี่ยงโรคหัวใจ (เกณฑ์ปกติ)</h2>
                    <div style="font-size: 18px; color:#E2E8F0; margin-bottom: 15px;">
                        ดัชนีความปลอดภัยสุขภาพหัวใจ: <span class="cyber-font" style="font-size:28px; color:#34D399; font-weight:700;">{safe_percent:.1f}%</span>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.7); border-radius: 10px; height: 14px; width: 100%; overflow: hidden; margin-bottom: 15px;">
                        <div style="background: linear-gradient(90deg, #10B981, #34D399); width: {safe_percent}%; height: 100%;"></div>
                    </div>
                    <p style="font-size: 14px; color: #CBD5E1; margin:0;">
                        คำแนะนำ: ค่าพารามิเตอร์สุขภาพของคุณอยู่ในเกณฑ์ปกติ (ประเมินจาก K={k_val}) แนะนำให้ออกกำลังกายอย่างสม่ำเสมอ ควบคุมอาหาร และตรวจสุขภาพประจำปี
                    </p>
                </div>
                """, unsafe_allow_html=True)

# ------------------------------------------
# TAB 2: หน้าสถิติและข้อมูลตัวอย่าง
# ------------------------------------------
with tab2:
    st.markdown("""
    <div class="glass-card">
        <span class="section-badge cyber-font">DATASET OVERVIEW</span>
        <h3 style="margin:0;">📊 สถิติชุดข้อมูลที่ใช้ฝึกสอนโมเดล (Heart3.csv)</h3>
    </div>
    """, unsafe_allow_html=True)
    
    total_cases = len(dt)
    disease_cases = int(dt['HeartDisease'].sum())
    normal_cases = total_cases - disease_cases
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <div style="color:#94A3B8; font-size:14px;">ข้อมูลทั้งหมดในระบบ</div>
            <div class="cyber-font" style="font-size:36px; font-weight:700; color:#38BDF8;">{total_cases:,}</div>
            <div style="color:#64748B; font-size:12px;">รายการ (Records)</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <div style="color:#94A3B8; font-size:14px;">กลุ่มเสี่ยง / เป็นโรคหัวใจ (1)</div>
            <div class="cyber-font" style="font-size:36px; font-weight:700; color:#FB7185;">{disease_cases:,}</div>
            <div style="color:#FB7185; font-size:12px;">คิดเป็น {(disease_cases/total_cases)*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <div style="color:#94A3B8; font-size:14px;">กลุ่มปกติ (0)</div>
            <div class="cyber-font" style="font-size:36px; font-weight:700; color:#34D399;">{normal_cases:,}</div>
            <div style="color:#34D399; font-size:12px;">คิดเป็น {(normal_cases/total_cases)*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("📈 กราฟเปรียบเทียบจำนวนผู้ป่วยในชุดข้อมูล")
    heart_counts = dt['HeartDisease'].value_counts().rename(
        index={0: "ไม่เป็นโรคหัวใจ (0)", 1: "เป็นโรคหัวใจ (1)"}
    )
    st.bar_chart(heart_counts, color="#38BDF8")
    
    st.subheader("🗂️ ตัวอย่างข้อมูล 10 แถวแรก")
    st.dataframe(dt.head(10), use_container_width=True)