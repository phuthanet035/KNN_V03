import os
import streamlit as st
import streamlit.components.v1 as components
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
    page_title="TITAN-X SINGULARITY | Quantum Cardiac Metaverse",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. SINGULARITY / TITAN-X METAVERSE CSS
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&family=Orbitron:wght@500;700;900&display=swap');

    /* พื้นหลังหลัก: Deep Cosmic Void + Quantum Grid + Aurora Nebula */
    .stApp {
        background-color: #02040A;
        background-image: 
            radial-gradient(ellipse at 15% 15%, rgba(6, 182, 212, 0.18) 0%, transparent 45%),
            radial-gradient(ellipse at 85% 20%, rgba(168, 85, 247, 0.16) 0%, transparent 45%),
            radial-gradient(ellipse at 50% 90%, rgba(16, 185, 129, 0.12) 0%, transparent 50%),
            linear-gradient(rgba(34, 211, 238, 0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(34, 211, 238, 0.04) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 100% 100%, 32px 32px, 32px 32px;
        color: #E2E8F0;
        font-family: 'Prompt', sans-serif;
    }

    /* Sidebar สไตล์ยานแม่ระดับ Titan */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(6, 11, 25, 0.95) 0%, rgba(2, 4, 10, 0.99) 100%);
        backdrop-filter: blur(28px);
        border-right: 1px solid rgba(34, 211, 238, 0.4);
        box-shadow: 10px 0 40px rgba(0, 0, 0, 0.85);
    }

    .orbitron {
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 2.2px;
    }

    /* แถบสถานะ Quantum Telemetry ด้านบนสุด */
    .titan-hud-bar {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(90deg, rgba(15, 23, 42, 0.9), rgba(49, 46, 129, 0.7), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(34, 211, 238, 0.45);
        border-radius: 14px;
        padding: 12px 24px;
        margin-bottom: 22px;
        box-shadow: 0 0 30px rgba(6, 182, 212, 0.2);
    }
    .quantum-dot {
        height: 11px; width: 11px;
        background-color: #10B981;
        border-radius: 50%;
        display: inline-block;
        margin-right: 10px;
        box-shadow: 0 0 14px #10B981;
        animation: quantumPulse 1.3s infinite;
    }
    @keyframes quantumPulse {
        0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.85); }
        70% { transform: scale(1.2); box-shadow: 0 0 0 14px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* เวที Singularity Core */
    .singularity-stage {
        position: relative;
        background: radial-gradient(circle at center, rgba(30, 41, 59, 0.88) 0%, rgba(5, 8, 20, 0.96) 100%);
        backdrop-filter: blur(30px);
        border: 1px solid rgba(139, 92, 246, 0.55);
        border-radius: 32px;
        padding: 38px 28px 28px 28px;
        text-align: center;
        margin-bottom: 28px;
        overflow: hidden;
        box-shadow: 0 0 70px rgba(34, 211, 238, 0.2), inset 0 0 50px rgba(168, 85, 247, 0.15);
    }

    /* ลำแสงสแกน 2 ชั้น */
    .plasma-beam {
        position: absolute;
        left: 0; width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, #22D3EE, #E879F9, #22D3EE, transparent);
        box-shadow: 0 0 20px #22D3EE, 0 0 35px #E879F9;
        animation: plasmaScan 4s ease-in-out infinite;
    }
    @keyframes plasmaScan {
        0%, 100% { top: 2%; opacity: 0.2; }
        50% { top: 96%; opacity: 1; }
    }

    /* วงแหวนพลังงาน 4 ชั้นหมุนรอบแกนหัวใจ */
    .titan-reactor {
        position: relative;
        width: 145px; height: 145px;
        margin: 0 auto 18px auto;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .ring-alpha {
        position: absolute; width: 145px; height: 145px;
        border: 2px dashed rgba(34, 211, 238, 0.7);
        border-radius: 50%;
        animation: rotCW 15s linear infinite;
    }
    .ring-beta {
        position: absolute; width: 120px; height: 120px;
        border: 2.5px solid transparent;
        border-top: 2.5px solid #E879F9;
        border-bottom: 2.5px solid #22D3EE;
        border-radius: 50%;
        animation: rotCCW 7s linear infinite;
    }
    .ring-gamma {
        position: absolute; width: 96px; height: 96px;
        border: 1.5px dotted rgba(52, 211, 153, 0.85);
        border-radius: 50%;
        animation: rotCW 4s linear infinite;
    }
    .core-organ {
        font-size: 50px;
        animation: titanBeat 1.05s ease-in-out infinite;
        filter: drop-shadow(0 0 22px rgba(244, 63, 94, 0.95));
    }
    @keyframes rotCW { 100% { transform: rotate(360deg); } }
    @keyframes rotCCW { 100% { transform: rotate(-360deg); } }
    @keyframes titanBeat {
        0%, 100% { transform: scale(1); }
        15% { transform: scale(1.28); }
        30% { transform: scale(1); }
        45% { transform: scale(1.18); }
    }

    .titan-title {
        font-size: 2.85rem;
        font-weight: 800;
        background: linear-gradient(90deg, #67E8F9 0%, #C084FC 50%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 6px 0 10px 0;
    }

    /* กล่องคอนโซลสไตล์ยานแม่ */
    .titan-panel {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.85), rgba(30, 41, 59, 0.45));
        backdrop-filter: blur(22px);
        border: 1px solid rgba(34, 211, 238, 0.35);
        border-left: 4px solid #22D3EE;
        border-radius: 18px;
        padding: 20px 24px;
        margin-bottom: 18px;
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.5);
    }

    /* การ์ดผลลัพธ์ระดับ Singularity */
    .singularity-danger {
        background: radial-gradient(circle at top right, rgba(225, 29, 72, 0.35), rgba(10, 15, 30, 0.96));
        border: 2px solid #FB7185;
        box-shadow: 0 0 50px rgba(244, 63, 94, 0.4), inset 0 0 35px rgba(244, 63, 94, 0.18);
        border-radius: 26px;
        padding: 30px;
    }
    .singularity-safe {
        background: radial-gradient(circle at top right, rgba(16, 185, 129, 0.35), rgba(10, 15, 30, 0.96));
        border: 2px solid #34D399;
        box-shadow: 0 0 50px rgba(16, 185, 129, 0.4), inset 0 0 35px rgba(16, 185, 129, 0.18);
        border-radius: 26px;
        padding: 30px;
    }

    /* ปุ่มสั่งการ Titan Overdrive */
    div.stFormSubmitButton > button {
        background: linear-gradient(90deg, #06B6D4 0%, #7C3AED 50%, #EC4899 100%) !important;
        background-size: 200% auto !important;
        color: #FFFFFF !important;
        font-family: 'Prompt', sans-serif !important;
        font-size: 21px !important;
        font-weight: 700 !important;
        letter-spacing: 1.2px;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        border-radius: 18px !important;
        padding: 20px 34px !important;
        width: 100%;
        box-shadow: 0 0 40px rgba(124, 58, 237, 0.6) !important;
        transition: all 0.35s ease !important;
    }
    div.stFormSubmitButton > button:hover {
        background-position: right center !important;
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 0 60px rgba(34, 211, 238, 0.95) !important;
    }

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
# 4. ฟังก์ชันสร้างแบบจำลองหัวใจ 3 มิติ (3D Parametric Heart)
# ==========================================
def create_3d_hologram_heart(is_danger=False):
    u = np.linspace(0, 2 * np.pi, 55)
    v = np.linspace(0, np.pi, 35)
    u, v = np.meshgrid(u, v)
    
    # สมการคณิตศาสตร์รูปทรงหัวใจ 3 มิติ
    x = 16 * (np.sin(u) ** 3) * np.sin(v)
    y = (13 * np.cos(u) - 5 * np.cos(2 * u) - 2 * np.cos(3 * u) - np.cos(4 * u)) * np.sin(v)
    z = 8 * np.cos(v)
    
    colorscale = [[0, '#881337'], [0.5, '#FB7185'], [1, '#FFE4E6']] if is_danger else [[0, '#064E3B'], [0.5, '#22D3EE'], [1, '#E0F2FE']]
    
    fig = go.Figure(data=[go.Surface(
        x=x, y=z, z=y,
        colorscale=colorscale,
        showscale=False,
        opacity=0.88,
        contours=dict(
            x=dict(show=True, color='rgba(255,255,255,0.18)', width=1),
            y=dict(show=True, color='rgba(255,255,255,0.18)', width=1),
            z=dict(show=True, color='rgba(255,255,255,0.18)', width=1)
        )
    )])
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.55, y=1.55, z=0.85))
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=310
    )
    return fig

# ==========================================
# 5. แถบควบคุมด้านข้าง (Titan-X Control Deck)
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 15px 0;">
        <div class="orbitron" style="font-size:24px; font-weight:900; color:#22D3EE;">TITAN-X</div>
        <div style="font-size:11px; color:#E879F9; letter-spacing:2.5px;">SINGULARITY BIOMETRIC AI</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### 🎛️ ควบคุมสมองกล KNN")
    
    k_val = st.slider(
        "จำนวนเพื่อนบ้านใกล้เคียง (K-Neighbors)",
        min_value=1, max_value=15, value=5, step=2
    )
    use_scaler = st.toggle(
        "ปรับสมดุลเวกเตอร์ (StandardScaler)",
        value=True
    )
    enable_voice = st.toggle(
        "🔊 เปิดเสียง AI พากย์รายงานผล",
        value=True,
        help="ให้ระบบอ่านสรุปผลการวิเคราะห์เป็นเสียงภาษาไทยอัตโนมัติ"
    )
    
    model, feature_cols, model_acc = train_knn_model(dt, k_val, use_scaler)
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(6, 182, 212, 0.2), rgba(168, 85, 247, 0.2)); 
                border: 1px solid rgba(34, 211, 238, 0.5); border-radius: 18px; padding: 18px; margin-top: 20px; text-align: center;">
        <div class="orbitron" style="font-size: 11px; color: #94A3B8;">NEURAL PRECISION</div>
        <div class="orbitron" style="font-size: 34px; font-weight: 900; color: #34D399; margin: 6px 0;">{model_acc*100:.2f}%</div>
        <div style="font-size: 12px; color: #CBD5E1;">ฐานข้อมูลอ้างอิง: <b>{len(dt):,}</b> เคส</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 6. แถบสถานะ HUD & เวที Singularity Core
# ==========================================
st.markdown(f"""
<div class="titan-hud-bar">
    <div><span class="quantum-dot"></span><b style="color:#34D399;">SINGULARITY CORE ONLINE</b> | อัลกอริทึม: <b>K-Nearest Neighbors (K={k_val})</b></div>
    <div class="orbitron" style="color:#22D3EE; font-size:12px;">3D HOLOGRAM • SYNAPSE TRACE • VOICE AI: {"ON" if enable_voice else "OFF"}</div>
</div>

<div class="singularity-stage">
    <div class="plasma-beam"></div>
    <div class="titan-reactor">
        <div class="ring-alpha"></div>
        <div class="ring-beta"></div>
        <div class="ring-gamma"></div>
        <div class="core-organ">🫀</div>
    </div>
    <div class="orbitron" style="color:#22D3EE; font-size:12px; letter-spacing:3.5px;">TITAN-X METAVERSE CARDIAC DIAGNOSTIC ENGINE</div>
    <div class="titan-title">ระบบวิเคราะห์หัวใจควอนตัมและจำลองโครงสร้าง 3 มิติ</div>
    <p style="color:#94A3B8; font-size:16px; max-width:780px; margin:0 auto;">
        ประมวลผลข้อมูลสุขภาพ 11 มิติ พร้อมสร้าง <b>โมเดลหัวใจ 3 มิติ</b>, <b>เส้นประสาทเชื่อมโยงเพื่อนบ้าน $K$ ลำดับ</b> และ <b>เสียง AI รายงานผล</b>
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 7. เมนูหลัก (Tabs)
# ==========================================
tab1, tab2 = st.tabs(["🛸 ห้องปฏิบัติการจำลอง 3 มิติ (Quantum Diagnostic)", "📡 คลังข้อมูลชีวภาพ (Telemetry Database)"])

with tab1:
    with st.form("prediction_form"):
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
            <div class="titan-panel">
                <div class="orbitron" style="color:#22D3EE; font-size:12px;">SECTOR 01 // BIOMETRIC PROFILE</div>
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
            <div class="titan-panel" style="border-left-color: #E879F9;">
                <div class="orbitron" style="color:#E879F9; font-size:12px;">SECTOR 02 // CLINICAL TELEMETRY</div>
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
        submitted = st.form_submit_button("🚀 เริ่มการสแกนระดับ SINGULARITY และสร้างโฮโลแกรม 3 มิติ")

    # ==========================================
    # แสดงผลลัพธ์สุดอลังการ (3D Heart + 3D Synapse Space + Voice AI)
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

        # ดึงเพื่อนบ้านที่ใกล้ที่สุด K ลำดับแรกมาแสดงผล
        if use_scaler:
            scaler_obj = model.named_steps['standardscaler']
            knn_obj = model.named_steps['kneighborsclassifier']
            distances, indices = knn_obj.kneighbors(scaler_obj.transform(x_input))
        else:
            distances, indices = model.kneighbors(x_input)

        nearest_neighbors_df = dt.iloc[indices[0]].copy()
        nearest_neighbors_df.insert(0, "ระยะห่างเวกเตอร์", np.round(distances[0], 3))

        # ระบบเสียง AI พากย์รายงานผลอัตโนมัติ (Web Speech API)
        if enable_voice:
            speech_text = (
                f"ระบบไททันเอ็กซ์สแกนเสร็จสิ้น ตรวจพบความเสี่ยงเป็นโรคหัวใจ {risk_percent:.0f} เปอร์เซ็นต์ แนะนำให้ปรึกษาแพทย์ผู้เชี่ยวชาญ"
                if prediction == 1 else
                f"ระบบไททันเอ็กซ์สแกนเสร็จสิ้น ไม่พบความเสี่ยงโรคหัวใจ ดัชนีความปลอดภัยสุขภาพหัวใจอยู่ที่ {safe_percent:.0f} เปอร์เซ็นต์"
            )
            components.html(f"""
            <script>
                window.speechSynthesis.cancel();
                const msg = new SpeechSynthesisUtterance("{speech_text}");
                msg.lang = 'th-TH';
                msg.rate = 1.05;
                window.speechSynthesis.speak(msg);
            </script>
            """, height=0)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # แถวที่ 1: โฮโลแกรมหัวใจ 3 มิติ + การ์ดผลวินิจฉัย + หน้าปัด Gauge
        h_col1, h_col2, h_col3 = st.columns([1, 1.4, 1], gap="medium")
        
        with h_col1:
            st.markdown("""
            <div class="titan-panel" style="text-align:center; padding:10px; margin-bottom:5px;">
                <span class="orbitron" style="font-size:11px; color:#22D3EE;">3D HOLOGRAPHIC ORGAN</span>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(create_3d_hologram_heart(is_danger=(prediction == 1)), use_container_width=True)

        with h_col2:
            if prediction == 1:
                st.markdown(f"""
                <div class="singularity-danger">
                    <div class="orbitron" style="color:#FB7185; font-size:12px; letter-spacing:2px;">🚨 SINGULARITY ALERT // CLASS 1</div>
                    <h2 style="color:#FDA4AF !important; margin:8px 0;">ตรวจพบความเสี่ยงโรคหัวใจ</h2>
                    <p style="font-size:16px; color:#E2E8F0;">
                        เพื่อนบ้านใกล้เคียง <b>{int(nearest_neighbors_df['HeartDisease'].sum())} จาก {k_val} คน</b> อยู่ในกลุ่มโรคหัวใจ
                        คิดเป็นความเสี่ยง <b style="color:#FB7185; font-size:24px;">{risk_percent:.1f}%</b>
                    </p>
                    <p style="font-size:13.5px; color:#CBD5E1; line-height:1.6;">
                        <b>คำแนะนำ:</b> ควรนำผลวิเคราะห์นี้ไปปรึกษาแพทย์ผู้เชี่ยวชาญเพื่อตรวจคลื่นไฟฟ้าหัวใจและวินิจฉัยอย่างละเอียด
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="singularity-safe">
                    <div class="orbitron" style="color:#34D399; font-size:12px; letter-spacing:2px;">🛡️ TITAN SHIELD ACTIVE // CLASS 0</div>
                    <h2 style="color:#6EE7B7 !important; margin:8px 0;">ไม่พบความเสี่ยงโรคหัวใจ</h2>
                    <p style="font-size:16px; color:#E2E8F0;">
                        เพื่อนบ้านใกล้เคียง <b>{k_val - int(nearest_neighbors_df['HeartDisease'].sum())} จาก {k_val} คน</b> อยู่ในกลุ่มสุขภาพปกติ
                        ดัชนีความปลอดภัย <b style="color:#34D399; font-size:24px;">{safe_percent:.1f}%</b>
                    </p>
                    <p style="font-size:13.5px; color:#CBD5E1; line-height:1.6;">
                        <b>คำแนะนำ:</b> ค่าสุขภาพของคุณอยู่ในเกณฑ์ดี แนะนำให้ออกกำลังกายสม่ำเสมอและตรวจสุขภาพประจำปี
                    </p>
                </div>
                """, unsafe_allow_html=True)

        with h_col3:
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_percent,
                number={'suffix': "%", 'font': {'size': 38, 'color': '#FB7185' if prediction == 1 else '#34D399', 'family': 'Orbitron'}},
                title={'text': "ดัชนีความเสี่ยง (RISK INDEX)", 'font': {'size': 14, 'color': '#E2E8F0', 'family': 'Prompt'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#94A3B8"},
                    'bar': {'color': "#FB7185" if prediction == 1 else "#34D399"},
                    'bgcolor': "rgba(15, 23, 42, 0.8)",
                    'borderwidth': 2,
                    'bordercolor': "rgba(34, 211, 238, 0.4)",
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(16, 185, 129, 0.2)'},
                        {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                        {'range': [70, 100], 'color': 'rgba(244, 63, 94, 0.25)'}
                    ],
                }
            ))
            gauge_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=15, r=15, t=45, b=10))
            st.plotly_chart(gauge_fig, use_container_width=True)

        # แถวที่ 2: อวกาศ 3 มิติพร้อมเส้นประสาทเชื่อมโยงเพื่อนบ้าน K ลำดับ + กราฟเรดาร์
        st.markdown("<br>", unsafe_allow_html=True)
        viz_c1, viz_c2 = st.columns([1.35, 1], gap="large")

        with viz_c1:
            st.markdown(f"""
            <div class="titan-panel" style="margin-bottom:8px; padding:14px 20px;">
                <div class="orbitron" style="color:#22D3EE; font-size:11px;">3D KNN NEURAL SYNAPSE MAPPING (K={k_val})</div>
                <h4 style="margin:2px 0 0 0;">🌌 แผนที่อวกาศ 3 มิติและเส้นเชื่อมโยงเพื่อนบ้าน {k_val} คนที่ใกล้คุณที่สุด</h4>
            </div>
            """, unsafe_allow_html=True)
            
            df_normal = dt[dt['HeartDisease'] == 0]
            df_disease = dt[dt['HeartDisease'] == 1]

            fig_3d = go.Figure()
            fig_3d.add_trace(go.Scatter3d(
                x=df_normal['Age'], y=df_normal['MaxHR'], z=df_normal['Cholesterol'],
                mode='markers', name='กลุ่มปกติ (0)',
                marker=dict(size=3.5, color='#34D399', opacity=0.35)
            ))
            fig_3d.add_trace(go.Scatter3d(
                x=df_disease['Age'], y=df_disease['MaxHR'], z=df_disease['Cholesterol'],
                mode='markers', name='กลุ่มโรคหัวใจ (1)',
                marker=dict(size=3.5, color='#FB7185', opacity=0.35)
            ))

            # วาดเส้นประสาทเชื่อมโยงจากจุดของผู้ใช้ไปยังเพื่อนบ้าน K ลำดับแรก
            for _, row in nearest_neighbors_df.iterrows():
                fig_3d.add_trace(go.Scatter3d(
                    x=[age, row['Age']], y=[max_hr, row['MaxHR']], z=[cholesterol, row['Cholesterol']],
                    mode='lines', showlegend=False,
                    line=dict(color='#E879F9', width=3, dash='dot')
                ))

            # จุดตำแหน่งของผู้ใช้
            fig_3d.add_trace(go.Scatter3d(
                x=[age], y=[max_hr], z=[cholesterol],
                mode='markers+text', name='ตำแหน่งของคุณ (YOU)',
                text=['📍 คุณ'], textposition='top center',
                marker=dict(size=12, color='#22D3EE', symbol='diamond', line=dict(color='white', width=2))
            ))
            fig_3d.update_layout(
                scene=dict(
                    xaxis=dict(title='อายุ (Age)', backgroundcolor='rgba(15,23,42,0.5)', gridcolor='rgba(34,211,238,0.2)'),
                    yaxis=dict(title='ชีพจรสูงสุด (MaxHR)', backgroundcolor='rgba(15,23,42,0.5)', gridcolor='rgba(34,211,238,0.2)'),
                    zaxis=dict(title='คอเลสเตอรอล', backgroundcolor='rgba(15,23,42,0.5)', gridcolor='rgba(34,211,238,0.2)'),
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
            <div class="titan-panel" style="border-left-color:#E879F9; margin-bottom:8px; padding:14px 20px;">
                <div class="orbitron" style="color:#E879F9; font-size:11px;">360° BIOMETRIC RADAR</div>
                <h4 style="margin:2px 0 0 0;">🕸️ เทียบค่าสุขภาพของคุณกับเกณฑ์ปกติ</h4>
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

        # แถวที่ 3: ตารางเปิดโปงข้อมูลเพื่อนบ้าน K ลำดับแรก (Explainable KNN)
        st.markdown(f"""
        <div class="titan-panel" style="margin-top:15px;">
            <div class="orbitron" style="color:#34D399; font-size:11px;">EXPLAINABLE KNN TRACE // NEAREST {k_val} NEIGHBORS</div>
            <h4 style="margin:4px 0 10px 0;">🔍 ข้อมูลผู้ป่วย {k_val} ลำดับแรกในฐานข้อมูลที่มีความใกล้เคียงกับคุณมากที่สุด</h4>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(nearest_neighbors_df, use_container_width=True)

with tab2:
    st.markdown("""
    <div class="titan-panel">
        <div class="orbitron" style="color:#22D3EE; font-size:12px;">DATABASE TELEMETRY</div>
        <h3 style="margin:4px 0 0 0;">📊 คลังข้อมูลผู้ป่วยโรคหัวใจ (Heart3.csv)</h3>
    </div>
    """, unsafe_allow_html=True)
    
    total_cases = len(dt)
    disease_cases = int(dt['HeartDisease'].sum())
    normal_cases = total_cases - disease_cases
    
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="titan-panel" style="text-align:center;"><div style="color:#94A3B8;">จำนวนเคสทั้งหมด</div><div class="orbitron" style="font-size:38px; font-weight:900; color:#22D3EE;">{total_cases:,}</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="titan-panel" style="text-align:center; border-left-color:#FB7185;"><div style="color:#94A3B8;">กลุ่มเสี่ยงโรคหัวใจ (1)</div><div class="orbitron" style="font-size:38px; font-weight:900; color:#FB7185;">{disease_cases:,}</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="titan-panel" style="text-align:center; border-left-color:#34D399;"><div style="color:#94A3B8;">กลุ่มสุขภาพปกติ (0)</div><div class="orbitron" style="font-size:38px; font-weight:900; color:#34D399;">{normal_cases:,}</div></div>""", unsafe_allow_html=True)

    heart_counts = dt['HeartDisease'].value_counts().rename(index={0: "ไม่เป็นโรคหัวใจ (0)", 1: "เป็นโรคหัวใจ (1)"})
    st.bar_chart(heart_counts, color="#22D3EE")
    st.dataframe(dt.head(10), use_container_width=True)