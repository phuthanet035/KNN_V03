import os
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
    page_title="NEXUS CARDIO-AI | ระบบวิเคราะห์หัวใจระดับควอนตัม",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. ULTRA-OVERDRIVE HOLOGRAPHIC CSS
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&family=Orbitron:wght@500;700;900&display=swap');

    /* พื้นหลังหลัก: Deep Space + Cyber Grid + Aurora Glow */
    .stApp {
        background-color: #050811;
        background-image: 
            radial-gradient(circle at 15% 15%, rgba(56, 189, 248, 0.14) 0%, transparent 40%),
            radial-gradient(circle at 85% 80%, rgba(139, 92, 246, 0.14) 0%, transparent 40%),
            linear-gradient(rgba(56, 189, 248, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(56, 189, 248, 0.03) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 40px 40px, 40px 40px;
        color: #E2E8F0;
        font-family: 'Prompt', sans-serif;
    }

    /* Sidebar สไตล์ยานอวกาศ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 15, 30, 0.92) 0%, rgba(5, 8, 17, 0.96) 100%);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(56, 189, 248, 0.3);
        box-shadow: 5px 0 30px rgba(0, 0, 0, 0.6);
    }

    .orbitron {
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 2px;
    }

    /* แถบสถานะ HUD ด้านบนสุด */
    .hud-topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 12px;
        padding: 10px 22px;
        margin-bottom: 20px;
        font-size: 13px;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.1);
    }
    .pulse-dot {
        height: 10px;
        width: 10px;
        background-color: #34D399;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        box-shadow: 0 0 10px #34D399;
        animation: pulseAnim 1.5s infinite;
    }
    @keyframes pulseAnim {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 10px rgba(52, 211, 153, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
    }

    /* Hero Holographic Stage */
    .holo-stage {
        position: relative;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 27, 75, 0.65) 100%);
        backdrop-filter: blur(24px);
        border: 1px solid rgba(99, 102, 241, 0.45);
        border-radius: 28px;
        padding: 40px 30px;
        text-align: center;
        margin-bottom: 30px;
        overflow: hidden;
        box-shadow: 0 0 50px rgba(56, 189, 248, 0.15), inset 0 0 30px rgba(139, 92, 246, 0.1);
    }

    /* ลำแสงเลเซอร์สแกนวิ่งขึ้นลง */
    .scanner-beam {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, transparent, #38BDF8, #A855F7, transparent);
        box-shadow: 0 0 15px #38BDF8;
        animation: scanMove 4s ease-in-out infinite;
    }
    @keyframes scanMove {
        0%, 100% { top: 0%; opacity: 0.2; }
        50% { top: 98%; opacity: 1; }
    }

    /* วงแหวนเรดาร์หมุน 3 ชั้น */
    .radar-core {
        position: relative;
        width: 110px;
        height: 110px;
        margin: 0 auto 18px auto;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .ring-1 {
        position: absolute;
        width: 110px; height: 110px;
        border: 2px dashed rgba(56, 189, 248, 0.6);
        border-radius: 50%;
        animation: spinRight 12s linear infinite;
    }
    .ring-2 {
        position: absolute;
        width: 88px; height: 88px;
        border: 2px solid transparent;
        border-top: 2px solid #A855F7;
        border-bottom: 2px solid #38BDF8;
        border-radius: 50%;
        animation: spinLeft 6s linear infinite;
    }
    .heart-core {
        font-size: 42px;
        animation: heartBeat 1.2s ease-in-out infinite;
        filter: drop-shadow(0 0 12px rgba(244, 63, 94, 0.8));
    }
    @keyframes spinRight { 100% { transform: rotate(360deg); } }
    @keyframes spinLeft { 100% { transform: rotate(-360deg); } }
    @keyframes heartBeat {
        0%, 100% { transform: scale(1); }
        15% { transform: scale(1.22); }
        30% { transform: scale(1); }
        45% { transform: scale(1.15); }
    }

    .ultra-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #67E8F9 0%, #818CF8 50%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(103, 232, 249, 0.25);
        margin: 5px 0 10px 0;
    }

    /* การ์ดแบบ Cyber Chamber มีมุมตัดเรืองแสง */
    .cyber-chamber {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.78), rgba(30, 41, 59, 0.45));
        backdrop-filter: blur(18px);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-left: 4px solid #38BDF8;
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }

    /* การ์ดผลลัพธ์แบบ Hyper-Alert & Hyper-Shield */
    .hyper-danger {
        background: radial-gradient(circle at top right, rgba(225, 29, 72, 0.28), rgba(15, 23, 42, 0.92));
        border: 2px solid #FB7185;
        box-shadow: 0 0 40px rgba(244, 63, 94, 0.3), inset 0 0 25px rgba(244, 63, 94, 0.15);
        border-radius: 24px;
        padding: 30px;
    }
    .hyper-safe {
        background: radial-gradient(circle at top right, rgba(16, 185, 129, 0.28), rgba(15, 23, 42, 0.92));
        border: 2px solid #34D399;
        box-shadow: 0 0 40px rgba(16, 185, 129, 0.3), inset 0 0 25px rgba(16, 185, 129, 0.15);
        border-radius: 24px;
        padding: 30px;
    }

    /* ปุ่มสั่งการระดับ Quantum */
    div.stFormSubmitButton > button {
        background: linear-gradient(90deg, #06B6D4 0%, #6366F1 50%, #EC4899 100%) !important;
        background-size: 200% auto !important;
        color: #FFFFFF !important;
        font-family: 'Prompt', sans-serif !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        letter-spacing: 1px;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 16px !important;
        padding: 18px 32px !important;
        width: 100%;
        box-shadow: 0 0 30px rgba(99, 102, 241, 0.5) !important;
        transition: all 0.35s ease !important;
    }
    div.stFormSubmitButton > button:hover {
        background-position: right center !important;
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 0 45px rgba(6, 182, 212, 0.85) !important;
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
# 4. แถบควบคุมด้านข้าง (Quantum Control Sidebar)
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 15px 0;">
        <div class="orbitron" style="font-size:22px; font-weight:900; color:#67E8F9;">NEXUS-AI</div>
        <div style="font-size:11px; color:#818CF8; letter-spacing:2px;">BIOMETRIC NEURAL CORE</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### 🎛️ ปรับแต่งพารามิเตอร์ KNN")
    
    k_val = st.slider(
        "จำนวนเพื่อนบ้านใกล้เคียง (K-Neighbors)",
        min_value=1, max_value=15, value=5, step=2
    )
    use_scaler = st.toggle(
        "ระบบปรับสมดุลเวกเตอร์ (StandardScaler)",
        value=True
    )
    
    model, feature_cols, model_acc = train_knn_model(dt, k_val, use_scaler)
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(6, 182, 212, 0.15), rgba(99, 102, 241, 0.15)); 
                border: 1px solid rgba(103, 232, 249, 0.4); border-radius: 16px; padding: 18px; margin-top: 20px; text-align: center;">
        <div class="orbitron" style="font-size: 11px; color: #94A3B8;">MODEL PRECISION</div>
        <div class="orbitron" style="font-size: 32px; font-weight: 900; color: #34D399; margin: 5px 0;">{model_acc*100:.2f}%</div>
        <div style="font-size: 12px; color: #CBD5E1;">ฐานข้อมูลฝึกสอน: <b>{len(dt):,}</b> เคส</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 5. แถบสถานะ HUD & เวที Holographic 3D
# ==========================================
st.markdown(f"""
<div class="hud-topbar">
    <div><span class="pulse-dot"></span><b style="color:#34D399;">SYSTEM ONLINE</b> | โหมดจำแนกข้อมูล: <b>K-Nearest Neighbors (K={k_val})</b></div>
    <div class="orbitron" style="color:#67E8F9; font-size:12px;">SCALER: {"ACTIVE" if use_scaler else "BYPASS"} • 11-DIMENSIONAL VECTOR</div>
</div>

<div class="holo-stage">
    <div class="scanner-beam"></div>
    <div class="radar-core">
        <div class="ring-1"></div>
        <div class="ring-2"></div>
        <div class="heart-core">🫀</div>
    </div>
    <div class="orbitron" style="color:#67E8F9; font-size:12px; letter-spacing:3px;">QUANTUM CARDIAC DIAGNOSTIC INTERFACE</div>
    <div class="ultra-title">ศูนย์วิเคราะห์และทำนายความเสี่ยงโรคหัวใจอัจฉริยะ</div>
    <p style="color:#94A3B8; font-size:16px; max-width:720px; margin:0 auto;">
        ป้อนค่าพารามิเตอร์ทางชีวภาพทั้ง 11 มิติ เพื่อให้ระบบ AI คำนวณระยะห่างเชิงเวกเตอร์และจำแนกความเสี่ยงแบบเรียลไทม์
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 6. เมนูหลัก (Tabs)
# ==========================================
tab1, tab2 = st.tabs(["🛸 ห้องปฏิบัติการทำนายผล (AI Diagnostic)", "📡 ฐานข้อมูลและสถิติเชิงลึก (Telemetry Data)"])

with tab1:
    with st.form("prediction_form"):
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
            <div class="cyber-chamber">
                <div class="orbitron" style="color:#38BDF8; font-size:12px;">MODULE 01 // BIOMETRIC PROFILE</div>
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
            <div class="cyber-chamber" style="border-left-color: #A855F7;">
                <div class="orbitron" style="color:#C084FC; font-size:12px;">MODULE 02 // CLINICAL TELEMETRY</div>
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
        submitted = st.form_submit_button("⚡ เริ่มการสแกนและทำนายผลด้วย AI ทันที")

    # ==========================================
    # แสดงผลลัพธ์พร้อม Radar Chart 360 องศา
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
        res_col1, res_col2 = st.columns([1.4, 1], gap="large")

        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="hyper-danger">
                    <div class="orbitron" style="color:#FB7185; font-size:13px; letter-spacing:2px;">⚠️ DIAGNOSTIC ALERT // CLASS 1</div>
                    <h2 style="color:#FDA4AF !important; margin:10px 0;">ตรวจพบความเสี่ยงเป็นโรคหัวใจ</h2>
                    <div style="font-size:18px; margin: 15px 0;">
                        ความน่าจะเป็นจากเพื่อนบ้าน {k_val} ลำดับแรก: 
                        <span class="orbitron" style="font-size:34px; font-weight:900; color:#FB7185;">{risk_percent:.1f}%</span>
                    </div>
                    <div style="background: rgba(5, 8, 17, 0.8); border-radius: 12px; height: 16px; width: 100%; overflow: hidden; margin-bottom: 18px; border:1px solid rgba(251,113,133,0.4);">
                        <div style="background: linear-gradient(90deg, #E11D48, #FB7185); width: {risk_percent}%; height: 100%;"></div>
                    </div>
                    <p style="font-size:14px; color:#E2E8F0; line-height:1.6;">
                        <b>ข้อแนะนำทางการแพทย์เบื้องต้น:</b> ค่าพารามิเตอร์ของคุณจัดอยู่ในกลุ่มใกล้เคียงกับผู้ป่วยโรคหัวใจ แนะนำให้นำผลตรวจนี้ไปปรึกษาแพทย์เฉพาะทางโรคหัวใจเพื่อตรวจคลื่นไฟฟ้าหัวใจและประเมินอย่างละเอียด
                    </p>
                </div>
                """, unsafe_allow_html=True)
                if os.path.exists("./img/yes.jpg"):
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.image("./img/yes.jpg", use_container_width=True)
            else:
                st.markdown(f"""
                <div class="hyper-safe">
                    <div class="orbitron" style="color:#34D399; font-size:13px; letter-spacing:2px;">🛡️ DIAGNOSTIC CLEAR // CLASS 0</div>
                    <h2 style="color:#6EE7B7 !important; margin:10px 0;">ไม่พบความเสี่ยงโรคหัวใจ (ปกติ)</h2>
                    <div style="font-size:18px; margin: 15px 0;">
                        ดัชนีความปลอดภัยสุขภาพหัวใจ: 
                        <span class="orbitron" style="font-size:34px; font-weight:900; color:#34D399;">{safe_percent:.1f}%</span>
                    </div>
                    <div style="background: rgba(5, 8, 17, 0.8); border-radius: 12px; height: 16px; width: 100%; overflow: hidden; margin-bottom: 18px; border:1px solid rgba(52,211,153,0.4);">
                        <div style="background: linear-gradient(90deg, #059669, #34D399); width: {safe_percent}%; height: 100%;"></div>
                    </div>
                    <p style="font-size:14px; color:#E2E8F0; line-height:1.6;">
                        <b>ข้อแนะนำทางการแพทย์เบื้องต้น:</b> ค่าพารามิเตอร์ของคุณอยู่ในเกณฑ์ปกติเมื่อเทียบกับฐานข้อมูล ควรรักษาสุขภาพด้วยการออกกำลังกายสม่ำเสมอ พักผ่อนให้เพียงพอ และตรวจสุขภาพประจำปี
                    </p>
                </div>
                """, unsafe_allow_html=True)
                if os.path.exists("./img/no.jpg"):
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.image("./img/no.jpg", use_container_width=True)

        # กราฟเรดาร์เปรียบเทียบค่าสุขภาพกับค่าเฉลี่ยกลุ่มปกติ
        with res_col2:
            st.markdown("""
            <div class="cyber-chamber" style="margin-bottom:10px; padding:15px;">
                <div class="orbitron" style="color:#67E8F9; font-size:12px;">BIOMETRIC RADAR COMPARISON</div>
                <h4 style="margin:2px 0 0 0;">🕸️ กราฟเปรียบเทียบค่าสุขภาพของคุณ</h4>
            </div>
            """, unsafe_allow_html=True)
            
            radar_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
            scaler_radar = MinMaxScaler()
            scaled_dt = pd.DataFrame(scaler_radar.fit_transform(dt[radar_cols]), columns=radar_cols)
            healthy_avg = scaled_dt[dt['HeartDisease'] == 0].mean().values
            user_scaled = scaler_radar.transform(x_input[radar_cols])[0]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=healthy_avg,
                theta=['อายุ', 'ความดันฯ', 'คอเลสเตอรอล', 'ชีพจรสูงสุด', 'Oldpeak'],
                fill='toself',
                name='ค่าเฉลี่ยกลุ่มปกติ',
                line_color='#34D399',
                fillcolor='rgba(52, 211, 153, 0.2)'
            ))
            fig.add_trace(go.Scatterpolar(
                r=user_scaled,
                theta=['อายุ', 'ความดันฯ', 'คอเลสเตอรอล', 'ชีพจรสูงสุด', 'Oldpeak'],
                fill='toself',
                name='ค่าของคุณ',
                line_color='#FB7185' if prediction == 1 else '#38BDF8',
                fillcolor='rgba(251, 113, 133, 0.35)' if prediction == 1 else 'rgba(56, 189, 248, 0.35)'
            ))
            fig.update_layout(
                polar=dict(
                    bgcolor='rgba(15, 23, 42, 0.6)',
                    radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor='rgba(148, 163, 184, 0.2)'),
                    angularaxis=dict(gridcolor='rgba(148, 163, 184, 0.2)', color='#E2E8F0')
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Prompt', color='#E2E8F0'),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                margin=dict(l=40, r=40, t=30, b=40),
                height=360
            )
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("""
    <div class="cyber-chamber">
        <div class="orbitron" style="color:#67E8F9; font-size:12px;">DATABASE TELEMETRY</div>
        <h3 style="margin:4px 0 0 0;">📊 สถิติชุดข้อมูลผู้ป่วยโรคหัวใจ (Heart3.csv)</h3>
    </div>
    """, unsafe_allow_html=True)
    
    total_cases = len(dt)
    disease_cases = int(dt['HeartDisease'].sum())
    normal_cases = total_cases - disease_cases
    
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="cyber-chamber" style="text-align:center;"><div style="color:#94A3B8;">จำนวนเคสทั้งหมด</div><div class="orbitron" style="font-size:36px; font-weight:900; color:#38BDF8;">{total_cases:,}</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="cyber-chamber" style="text-align:center; border-left-color:#FB7185;"><div style="color:#94A3B8;">กลุ่มเสี่ยงโรคหัวใจ (1)</div><div class="orbitron" style="font-size:36px; font-weight:900; color:#FB7185;">{disease_cases:,}</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="cyber-chamber" style="text-align:center; border-left-color:#34D399;"><div style="color:#94A3B8;">กลุ่มสุขภาพปกติ (0)</div><div class="orbitron" style="font-size:36px; font-weight:900; color:#34D399;">{normal_cases:,}</div></div>""", unsafe_allow_html=True)

    heart_counts = dt['HeartDisease'].value_counts().rename(index={0: "ไม่เป็นโรคหัวใจ (0)", 1: "เป็นโรคหัวใจ (1)"})
    st.bar_chart(heart_counts, color="#67E8F9")
    st.dataframe(dt.head(10), use_container_width=True)