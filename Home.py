import os
import base64
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================================
# 1. ตั้งค่าหน้าเว็บ (Page Configuration)
# ==========================================
st.set_page_config(
    page_title="OMEGA CARDIO-AI | Quantum Biometric Console",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ฟังก์ชันช่วยแปลงรูปภาพเป็น Base64 เพื่อล็อกขนาดและใส่กรอบ CSS สุดเท่
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

# ==========================================
# 2. GOD-TIER HOLOGRAPHIC & SCI-FI HUD CSS
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&family=Orbitron:wght@500;700;900&display=swap');

    /* พื้นหลังหลัก: Deep Obsidian + Bioluminescent Nebula + Cyber Matrix Grid */
    .stApp {
        background-color: #03060E;
        background-image: 
            radial-gradient(ellipse at 10% 10%, rgba(6, 182, 212, 0.16) 0%, transparent 45%),
            radial-gradient(ellipse at 90% 90%, rgba(139, 92, 246, 0.16) 0%, transparent 45%),
            radial-gradient(circle at 50% 50%, rgba(16, 185, 129, 0.06) 0%, transparent 60%),
            linear-gradient(rgba(56, 189, 248, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(56, 189, 248, 0.035) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 100% 100%, 36px 36px, 36px 36px;
        color: #E2E8F0;
        font-family: 'Prompt', sans-serif;
    }

    /* Sidebar สไตล์ห้องควบคุมยานแม่ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(7, 12, 26, 0.94) 0%, rgba(3, 6, 14, 0.98) 100%);
        backdrop-filter: blur(24px);
        border-right: 1px solid rgba(56, 189, 248, 0.35);
        box-shadow: 8px 0 35px rgba(0, 0, 0, 0.75);
    }

    .orbitron {
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 2px;
    }

    /* แถบสถานะ HUD ด้านบนสุด */
    .omega-hud-bar {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(90deg, rgba(15, 23, 42, 0.85), rgba(30, 27, 75, 0.75), rgba(15, 23, 42, 0.85));
        border: 1px solid rgba(56, 189, 248, 0.4);
        border-radius: 14px;
        padding: 12px 24px;
        margin-bottom: 22px;
        box-shadow: 0 0 25px rgba(6, 182, 212, 0.15);
    }
    .status-beacon {
        height: 10px; width: 10px;
        background-color: #10B981;
        border-radius: 50%;
        display: inline-block;
        margin-right: 10px;
        box-shadow: 0 0 12px #10B981;
        animation: beaconPulse 1.4s infinite;
    }
    @keyframes beaconPulse {
        0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.8); }
        70% { transform: scale(1.15); box-shadow: 0 0 0 12px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* เวที Holographic 3D Core */
    .omega-stage {
        position: relative;
        background: radial-gradient(circle at center, rgba(30, 41, 59, 0.85) 0%, rgba(9, 13, 28, 0.95) 100%);
        backdrop-filter: blur(28px);
        border: 1px solid rgba(99, 102, 241, 0.5);
        border-radius: 30px;
        padding: 42px 30px 32px 30px;
        text-align: center;
        margin-bottom: 30px;
        overflow: hidden;
        box-shadow: 0 0 60px rgba(56, 189, 248, 0.18), inset 0 0 40px rgba(139, 92, 246, 0.12);
    }

    /* ลำแสงเลเซอร์สแกน 2 ทิศทาง */
    .laser-sweep {
        position: absolute;
        left: 0; width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, #22D3EE, #C084FC, #22D3EE, transparent);
        box-shadow: 0 0 18px #22D3EE;
        animation: laserVertical 4.5s ease-in-out infinite;
    }
    @keyframes laserVertical {
        0%, 100% { top: 2%; opacity: 0.15; }
        50% { top: 96%; opacity: 1; }
    }

    /* วงแหวนพลังงาน 4 ชั้นหมุนรอบหัวใจ */
    .orbital-system {
        position: relative;
        width: 135px; height: 135px;
        margin: 0 auto 20px auto;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .orbit-1 {
        position: absolute; width: 135px; height: 135px;
        border: 2px dashed rgba(34, 211, 238, 0.65);
        border-radius: 50%;
        animation: spinCW 14s linear infinite;
    }
    .orbit-2 {
        position: absolute; width: 112px; height: 112px;
        border: 2px solid transparent;
        border-top: 2px solid #A855F7;
        border-bottom: 2px solid #22D3EE;
        border-radius: 50%;
        animation: spinCCW 7s linear infinite;
    }
    .orbit-3 {
        position: absolute; width: 90px; height: 90px;
        border: 1.5px dotted rgba(244, 114, 182, 0.75);
        border-radius: 50%;
        animation: spinCW 4s linear infinite;
    }
    .bio-heart {
        font-size: 46px;
        animation: cardiacPulse 1.1s ease-in-out infinite;
        filter: drop-shadow(0 0 18px rgba(244, 63, 94, 0.9));
    }
    @keyframes spinCW { 100% { transform: rotate(360deg); } }
    @keyframes spinCCW { 100% { transform: rotate(-360deg); } }
    @keyframes cardiacPulse {
        0%, 100% { transform: scale(1); }
        14% { transform: scale(1.26); }
        28% { transform: scale(1); }
        42% { transform: scale(1.18); }
    }

    /* คลื่นหัวใจ ECG วิ่งต่อเนื่อง */
    .ecg-wave-path {
        stroke-dasharray: 600;
        stroke-dashoffset: 600;
        animation: ecgFlow 2.8s linear infinite;
    }
    @keyframes ecgFlow {
        to { stroke-dashoffset: 0; }
    }

    .omega-title {
        font-size: 2.75rem;
        font-weight: 800;
        background: linear-gradient(90deg, #67E8F9 0%, #A5B4FC 45%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 6px 0 10px 0;
    }

    /* กล่องคอนโซลตัดมุมสไตล์ยานอวกาศ */
    .sci-fi-panel {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.82), rgba(30, 41, 59, 0.45));
        backdrop-filter: blur(20px);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-left: 4px solid #22D3EE;
        border-radius: 18px;
        padding: 20px 24px;
        margin-bottom: 18px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.45);
    }

    /* การ์ดผลลัพธ์ระดับ Hyper-Core (จัดวางรูปภาพคู่ข้อความแบบล็อกขนาด) */
    .omega-danger-box {
        background: radial-gradient(circle at top right, rgba(225, 29, 72, 0.32), rgba(10, 15, 30, 0.95));
        border: 2px solid #FB7185;
        box-shadow: 0 0 45px rgba(244, 63, 94, 0.35), inset 0 0 30px rgba(244, 63, 94, 0.15);
        border-radius: 24px;
        padding: 24px;
        display: flex;
        align-items: center;
        gap: 24px;
        min-height: 290px;
    }
    .omega-safe-box {
        background: radial-gradient(circle at top right, rgba(16, 185, 129, 0.32), rgba(10, 15, 30, 0.95));
        border: 2px solid #34D399;
        box-shadow: 0 0 45px rgba(16, 185, 129, 0.35), inset 0 0 30px rgba(16, 185, 129, 0.15);
        border-radius: 24px;
        padding: 24px;
        display: flex;
        align-items: center;
        gap: 24px;
        min-height: 290px;
    }

    /* กรอบรูปภาพแบบล็อกขนาดคงที่ 230x230px มองเห็นชัดเจนไม่ล้นจอ */
    .locked-img-frame-danger {
        flex-shrink: 0;
        width: 230px;
        height: 230px;
        border-radius: 18px;
        border: 2px solid #FB7185;
        box-shadow: 0 0 25px rgba(251, 113, 133, 0.45);
        overflow: hidden;
        background: #0F172A;
    }
    .locked-img-frame-safe {
        flex-shrink: 0;
        width: 230px;
        height: 230px;
        border-radius: 18px;
        border: 2px solid #34D399;
        box-shadow: 0 0 25px rgba(52, 211, 153, 0.45);
        overflow: hidden;
        background: #0F172A;
    }
    .locked-img-frame-danger img, .locked-img-frame-safe img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        object-position: center top;
        display: block;
    }

    /* ปุ่มสั่งการ Quantum Overdrive */
    div.stFormSubmitButton > button {
        background: linear-gradient(90deg, #06B6D4 0%, #6366F1 50%, #EC4899 100%) !important;
        background-size: 200% auto !important;
        color: #FFFFFF !important;
        font-family: 'Prompt', sans-serif !important;
        font-size: 21px !important;
        font-weight: 700 !important;
        letter-spacing: 1px;
        border: 1px solid rgba(255, 255, 255, 0.35) !important;
        border-radius: 16px !important;
        padding: 18px 32px !important;
        width: 100%;
        box-shadow: 0 0 35px rgba(99, 102, 241, 0.55) !important;
        transition: all 0.35s ease !important;
    }
    div.stFormSubmitButton > button:hover {
        background-position: right center !important;
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 0 55px rgba(34, 211, 238, 0.9) !important;
    }

    /* ปรับสีตัวอักษรให้อ่านง่ายสบายตา */
    label, .stMarkdown p { color: #CBD5E1 !important; }
    h1, h2, h3, h4 { color: #F8FAFC !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. โหลดข้อมูลและสร้างโมเดล KNN
# ==========================================
@st.cache_data
def load_data():
    file_path = "./data/Heart3.csv"
    if not os.path.exists(file_path):
        st.error(f"❌ ไม่พบไฟล์ข้อมูลที่ `{file_path}` กรุณาตรวจสอบโฟลเดอร์ data/")
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
# 4. แถบควบคุมด้านข้าง (Omega Control Deck)
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 15px 0;">
        <div class="orbitron" style="font-size:24px; font-weight:900; color:#22D3EE;">OMEGA-AI</div>
        <div style="font-size:11px; color:#A855F7; letter-spacing:2.5px;">QUANTUM NEURAL ENGINE</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### 🎛️ ควบคุมพารามิเตอร์ KNN")
    
    k_val = st.slider(
        "จำนวนเพื่อนบ้านใกล้เคียง (K-Neighbors)",
        min_value=1, max_value=15, value=5, step=2
    )
    use_scaler = st.toggle(
        "ปรับสมดุลเวกเตอร์ (StandardScaler)",
        value=True
    )
    
    model, feature_cols, model_acc = train_knn_model(dt, k_val, use_scaler)
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(6, 182, 212, 0.18), rgba(139, 92, 246, 0.18)); 
                border: 1px solid rgba(34, 211, 238, 0.45); border-radius: 18px; padding: 18px; margin-top: 20px; text-align: center;">
        <div class="orbitron" style="font-size: 11px; color: #94A3B8;">NEURAL ACCURACY</div>
        <div class="orbitron" style="font-size: 34px; font-weight: 900; color: #34D399; margin: 6px 0;">{model_acc*100:.2f}%</div>
        <div style="font-size: 12px; color: #CBD5E1;">ฐานข้อมูลอ้างอิง: <b>{len(dt):,}</b> เคส</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 5. แถบสถานะ HUD & เวที Holographic 3D Core
# ==========================================
st.markdown(f"""
<div class="omega-hud-bar">
    <div><span class="status-beacon"></span><b style="color:#34D399;">NEURAL CORE ONLINE</b> | อัลกอริทึม: <b>K-Nearest Neighbors (K={k_val})</b></div>
    <div class="orbitron" style="color:#22D3EE; font-size:12px;">VECTOR SCALING: {"ON" if use_scaler else "OFF"} • 3D SPACE MAPPING READY</div>
</div>

<div class="omega-stage">
    <div class="laser-sweep"></div>
    <div class="orbital-system">
        <div class="orbit-1"></div>
        <div class="orbit-2"></div>
        <div class="orbit-3"></div>
        <div class="bio-heart">🫀</div>
    </div>
    <div class="orbitron" style="color:#22D3EE; font-size:12px; letter-spacing:3.5px;">OMEGA QUANTUM CARDIAC DIAGNOSTIC CONSOLE</div>
    <div class="omega-title">ระบบวิเคราะห์และจำลองความเสี่ยงโรคหัวใจ 3 มิติ</div>
    <p style="color:#94A3B8; font-size:16px; max-width:760px; margin:0 auto 16px auto;">
        จำแนกข้อมูลสุขภาพเชิงลึก 11 มิติด้วย <b>K-Nearest Neighbors (KNN)</b> พร้อมจำลองตำแหน่งของคุณในอวกาศข้อมูล 3 มิติแบบเรียลไทม์
    </p>
    <svg width="340" height="45" viewBox="0 0 340 45" style="margin: 0 auto; display: block;">
        <path class="ecg-wave-path" fill="none" stroke="#22D3EE" stroke-width="2.5"
              d="M0,25 L70,25 L85,10 L96,38 L110,2 L125,43 L136,25 L190,25 L205,12 L216,36 L228,4 L240,40 L250,25 L340,25" />
    </svg>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 6. เมนูหลัก (Tabs)
# ==========================================
tab1, tab2 = st.tabs(["🛸 ห้องสแกนและจำลอง 3 มิติ (AI Diagnostic & 3D Space)", "📡 คลังข้อมูลชีวภาพ (Telemetry Database)"])

with tab1:
    with st.form("prediction_form"):
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
            <div class="sci-fi-panel">
                <div class="orbitron" style="color:#22D3EE; font-size:12px;">SECTOR 01 // BIOMETRIC INPUT</div>
                <h4 style="margin:4px 0 0 0;">🧬 ข้อมูลพื้นฐานและอาการทางกายภาพ</h4>
            </div>
            """, unsafe_allow_html=True)
            
            age = st.slider("1. อายุ (Age - ปี)", min_value=20, max_value=100, value=50)
            sex = st.selectbox("2. เพศ (Sex)", options=[1, 0], format_func=lambda x: "ชาย (1)" if x == 1 else "หญิง (0)")
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
            exercise_angina = st.selectbox("4. เจ็บหน้าอกขณะออกกำลังกาย (ExerciseAngina)", options=[0, 1], format_func=lambda x: "ไม่เจ็บ (0)" if x == 0 else "เจ็บ (1)")
            resting_bp = st.number_input("5. ความดันโลหิตขณะพัก (RestingBP - mm Hg)", min_value=0, max_value=250, value=130)
            cholesterol = st.number_input("6. ระดับคอเลสเตอรอล (Cholesterol - mg/dl)", min_value=0, max_value=700, value=200)

        with col2:
            st.markdown("""
            <div class="sci-fi-panel" style="border-left-color: #A855F7;">
                <div class="orbitron" style="color:#C084FC; font-size:12px;">SECTOR 02 // CLINICAL TELEMETRY</div>
                <h4 style="margin:4px 0 0 0;">📈 ผลตรวจทางคลินิกและคลื่นไฟฟ้าหัวใจ</h4>
            </div>
            """, unsafe_allow_html=True)
            
            fasting_bs = st.selectbox("7. น้ำตาลในเลือดหลังอดอาหาร > 120 mg/dl (FastingBS)", options=[0, 1], format_func=lambda x: "ปกติ <= 120 mg/dl (0)" if x == 0 else "สูง > 120 mg/dl (1)")
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
        submitted = st.form_submit_button("⚡ เริ่มการสแกนควอนตัมและสร้างแบบจำลอง 3 มิติ")

    # ==========================================
    # แสดงผลลัพธ์สุดอลังการ (ล็อกขนาดรูปภาพ + ผลวินิจฉัย + Gauge)
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
        
        # แถวที่ 1: การ์ดผลวินิจฉัยพร้อมรูปภาพล็อกขนาด 230x230px + หน้าปัด Gauge
        top_c1, top_c2 = st.columns([1.65, 1], gap="large")
        
        with top_c1:
            if prediction == 1:
                img_b64 = get_base64_image("./img/yes.jpg")
                img_html = f'<div class="locked-img-frame-danger"><img src="data:image/jpeg;base64,{img_b64}" alt="High Risk"></div>' if img_b64 else '<div class="locked-img-frame-danger" style="display:flex;align-items:center;justify-content:center;font-size:80px;">🚨</div>'
                
                st.markdown(f"""
                <div class="omega-danger-box">
                    {img_html}
                    <div>
                        <div class="orbitron" style="color:#FB7185; font-size:12px; letter-spacing:2px;">🚨 CRITICAL BIOMETRIC ALERT // CLASS 1</div>
                        <h2 style="color:#FDA4AF !important; margin:8px 0;">ตรวจพบความเสี่ยงเป็นโรคหัวใจ</h2>
                        <p style="font-size:16px; color:#E2E8F0; margin-bottom:10px;">
                            โมเดล KNN (K={k_val}) ประเมินแนวโน้มความเสี่ยงอยู่ที่ <b style="color:#FB7185; font-size:24px;">{risk_percent:.1f}%</b>
                        </p>
                        <p style="font-size:13.5px; color:#CBD5E1; line-height:1.55; margin:0;">
                            <b>คำแนะนำ:</b> ค่าพารามิเตอร์ของคุณอยู่ในโซนใกล้เคียงกับกลุ่มผู้ป่วยโรคหัวใจในฐานข้อมูล ควรปรึกษาแพทย์เฉพาะทางโรคหัวใจเพื่อตรวจคลื่นไฟฟ้าหัวใจอย่างละเอียด
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                img_b64 = get_base64_image("./img/no.jpg")
                img_html = f'<div class="locked-img-frame-safe"><img src="data:image/jpeg;base64,{img_b64}" alt="Low Risk"></div>' if img_b64 else '<div class="locked-img-frame-safe" style="display:flex;align-items:center;justify-content:center;font-size:80px;">🛡️</div>'
                
                st.markdown(f"""
                <div class="omega-safe-box">
                    {img_html}
                    <div>
                        <div class="orbitron" style="color:#34D399; font-size:12px; letter-spacing:2px;">🛡️ OPTIMAL CARDIAC SHIELD // CLASS 0</div>
                        <h2 style="color:#6EE7B7 !important; margin:8px 0;">ไม่พบความเสี่ยงโรคหัวใจ (เกณฑ์ปกติ)</h2>
                        <p style="font-size:16px; color:#E2E8F0; margin-bottom:10px;">
                            โมเดล KNN (K={k_val}) ประเมินดัชนีความปลอดภัยสุขภาพหัวใจอยู่ที่ <b style="color:#34D399; font-size:24px;">{safe_percent:.1f}%</b>
                        </p>
                        <p style="font-size:13.5px; color:#CBD5E1; line-height:1.55; margin:0;">
                            <b>คำแนะนำ:</b> ค่าพารามิเตอร์ของคุณอยู่ในโซนสุขภาพดี แนะนำให้ออกกำลังกายสม่ำเสมอ ควบคุมระดับคอเลสเตอรอลและความดันโลหิตให้คงที่
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with top_c2:
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_percent,
                number={'suffix': "%", 'font': {'size': 40, 'color': '#FB7185' if prediction == 1 else '#34D399', 'family': 'Orbitron'}},
                title={'text': "ดัชนีความเสี่ยงโรคหัวใจ (RISK SCORE)", 'font': {'size': 14, 'color': '#E2E8F0', 'family': 'Prompt'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                    'bar': {'color': "#FB7185" if prediction == 1 else "#34D399"},
                    'bgcolor': "rgba(15, 23, 42, 0.8)",
                    'borderwidth': 2,
                    'bordercolor': "rgba(56, 189, 248, 0.4)",
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(16, 185, 129, 0.2)'},
                        {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                        {'range': [70, 100], 'color': 'rgba(244, 63, 94, 0.25)'}
                    ],
                }
            ))
            gauge_fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': "#E2E8F0", 'family': "Prompt"},
                height=290,
                margin=dict(l=20, r=20, t=45, b=10)
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

        # แถวที่ 2: กราฟ 3D KNN Decision Space + กราฟเรดาร์ 360 องศา
        st.markdown("<br>", unsafe_allow_html=True)
        viz_c1, viz_c2 = st.columns([1.3, 1], gap="large")

        with viz_c1:
            st.markdown("""
            <div class="sci-fi-panel" style="margin-bottom:8px; padding:14px 20px;">
                <div class="orbitron" style="color:#22D3EE; font-size:11px;">3D QUANTUM VECTOR SPACE (หมุนและซูมได้)</div>
                <h4 style="margin:2px 0 0 0;">🌌 ตำแหน่งของคุณท่ามกลางฐานข้อมูล 3 มิติ</h4>
            </div>
            """, unsafe_allow_html=True)
            
            df_normal = dt[dt['HeartDisease'] == 0]
            df_disease = dt[dt['HeartDisease'] == 1]

            fig_3d = go.Figure()
            fig_3d.add_trace(go.Scatter3d(
                x=df_normal['Age'], y=df_normal['MaxHR'], z=df_normal['Cholesterol'],
                mode='markers', name='กลุ่มปกติ (0)',
                marker=dict(size=4, color='#34D399', opacity=0.45)
            ))
            fig_3d.add_trace(go.Scatter3d(
                x=df_disease['Age'], y=df_disease['MaxHR'], z=df_disease['Cholesterol'],
                mode='markers', name='กลุ่มโรคหัวใจ (1)',
                marker=dict(size=4, color='#FB7185', opacity=0.45)
            ))
            fig_3d.add_trace(go.Scatter3d(
                x=[age], y=[max_hr], z=[cholesterol],
                mode='markers+text', name='ตำแหน่งของคุณ (YOU)',
                text=['📍 คุณอยู่ตรงนี้'], textposition='top center',
                marker=dict(size=12, color='#22D3EE', symbol='diamond', line=dict(color='white', width=2))
            ))
            fig_3d.update_layout(
                scene=dict(
                    xaxis=dict(title='อายุ (Age)', backgroundcolor='rgba(15,23,42,0.5)', gridcolor='rgba(56,189,248,0.2)'),
                    yaxis=dict(title='ชีพจรสูงสุด (MaxHR)', backgroundcolor='rgba(15,23,42,0.5)', gridcolor='rgba(56,189,248,0.2)'),
                    zaxis=dict(title='คอเลสเตอรอล', backgroundcolor='rgba(15,23,42,0.5)', gridcolor='rgba(56,189,248,0.2)'),
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Prompt', color='#E2E8F0'),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                margin=dict(l=0, r=0, t=10, b=10),
                height=420
            )
            st.plotly_chart(fig_3d, use_container_width=True)

        with viz_c2:
            st.markdown("""
            <div class="sci-fi-panel" style="border-left-color:#A855F7; margin-bottom:8px; padding:14px 20px;">
                <div class="orbitron" style="color:#C084FC; font-size:11px;">360° BIOMETRIC RADAR</div>
                <h4 style="margin:2px 0 0 0;">🕸️ เทียบค่าของคุณกับเกณฑ์ปกติ</h4>
            </div>
            """, unsafe_allow_html=True)
            
            radar_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
            scaler_radar = MinMaxScaler()
            scaled_dt = pd.DataFrame(scaler_radar.fit_transform(dt[radar_cols]), columns=radar_cols)
            healthy_avg = scaled_dt[dt['HeartDisease'] == 0].mean().values
            user_scaled = scaler_radar.transform(x_input[radar_cols])[0]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=healthy_avg,
                theta=['อายุ', 'ความดันฯ', 'คอเลสเตอรอล', 'ชีพจรสูงสุด', 'Oldpeak'],
                fill='toself', name='ค่าเฉลี่ยกลุ่มปกติ',
                line_color='#34D399', fillcolor='rgba(52, 211, 153, 0.2)'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=user_scaled,
                theta=['อายุ', 'ความดันฯ', 'คอเลสเตอรอล', 'ชีพจรสูงสุด', 'Oldpeak'],
                fill='toself', name='ค่าของคุณ',
                line_color='#FB7185' if prediction == 1 else '#22D3EE',
                fillcolor='rgba(251, 113, 133, 0.35)' if prediction == 1 else 'rgba(34, 211, 238, 0.35)'
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor='rgba(15, 23, 42, 0.6)',
                    radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor='rgba(148, 163, 184, 0.2)'),
                    angularaxis=dict(gridcolor='rgba(148, 163, 184, 0.2)', color='#E2E8F0')
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Prompt', color='#E2E8F0'),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                margin=dict(l=35, r=35, t=25, b=25),
                height=420
            )
            st.plotly_chart(fig_radar, use_container_width=True)

with tab2:
    st.markdown("""
    <div class="sci-fi-panel">
        <div class="orbitron" style="color:#22D3EE; font-size:12px;">DATABASE TELEMETRY</div>
        <h3 style="margin:4px 0 0 0;">📊 คลังข้อมูลผู้ป่วยโรคหัวใจ (Heart3.csv)</h3>
    </div>
    """, unsafe_allow_html=True)
    
    total_cases = len(dt)
    disease_cases = int(dt['HeartDisease'].sum())
    normal_cases = total_cases - disease_cases
    
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="sci-fi-panel" style="text-align:center;"><div style="color:#94A3B8;">จำนวนเคสทั้งหมด</div><div class="orbitron" style="font-size:38px; font-weight:900; color:#22D3EE;">{total_cases:,}</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="sci-fi-panel" style="text-align:center; border-left-color:#FB7185;"><div style="color:#94A3B8;">กลุ่มเสี่ยงโรคหัวใจ (1)</div><div class="orbitron" style="font-size:38px; font-weight:900; color:#FB7185;">{disease_cases:,}</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="sci-fi-panel" style="text-align:center; border-left-color:#34D399;"><div style="color:#94A3B8;">กลุ่มสุขภาพปกติ (0)</div><div class="orbitron" style="font-size:38px; font-weight:900; color:#34D399;">{normal_cases:,}</div></div>""", unsafe_allow_html=True)

    heart_counts = dt['HeartDisease'].value_counts().rename(index={0: "ไม่เป็นโรคหัวใจ (0)", 1: "เป็นโรคหัวใจ (1)"})
    st.bar_chart(heart_counts, color="#22D3EE")
    st.dataframe(dt.head(10), use_container_width=True)