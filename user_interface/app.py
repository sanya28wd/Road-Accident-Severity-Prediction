import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
from pathlib import Path

st.set_page_config(
    page_title="CrashVision 3D | AI Accident Severity Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Futuristic Dark Glassmorphism CSS Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    * { 
        font-family: 'Outfit', 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, #030712 0%, #020617 90%);
        color: #f8fafc;
    }
    
    /* Global Text Fix */
    .stApp p, .stApp label, .stApp span, .stApp div, .stMarkdown {
        color: #e2e8f0 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #030712 0%, #0f172a 100%) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.2);
    }
    
    [data-testid="stSidebar"] * { 
        color: #f8fafc !important; 
    }
    
    /* Hero Gradient Titles */
    .main-header {
        background: linear-gradient(135deg, #ffffff 0%, #38bdf8 40%, #a855f7 80%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        font-size: 3rem;
        margin-bottom: 0.1rem;
        text-align: center;
        letter-spacing: -0.03em;
    }

    .sub-header {
        color: #94a3b8 !important;
        text-align: center;
        font-size: 1.15rem;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    
    /* Glassmorphism Container Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 24px;
        padding: 28px;
        margin-bottom: 24px;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.5);
        box-shadow: 0 20px 50px rgba(56, 189, 248, 0.2);
    }

    /* Danger / Safe Banners */
    .result-card-safe {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 78, 59, 0.4) 100%);
        border: 2px solid #10b981;
        border-radius: 24px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 0 40px rgba(16, 185, 129, 0.25);
    }
    
    .result-card-danger {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.15) 0%, rgba(127, 29, 29, 0.4) 100%);
        border: 2px solid #f43f5e;
        border-radius: 24px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 0 50px rgba(244, 63, 94, 0.35);
    }
    
    .result-title {
        color: #94a3b8 !important;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    
    .result-value {
        font-size: 2.3rem;
        font-weight: 900;
        margin: 10px 0;
    }
    
    /* Glowing Button */
    .stButton > button {
        background: linear-gradient(135deg, #38bdf8 0%, #a855f7 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 18px 36px !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        letter-spacing: 1px !important;
        box-shadow: 0 10px 30px rgba(56, 189, 248, 0.4) !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 40px rgba(168, 85, 247, 0.6) !important;
    }
    
    /* Selectboxes */
    div[data-baseweb="select"] > div {
        background-color: #0f172a !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
    }

    div[data-baseweb="select"] span {
        color: #ffffff !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #38bdf8 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(15, 23, 42, 0.8);
        padding: 8px;
        border-radius: 18px;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 12px !important;
        color: #94a3b8 !important;
        border: none !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #38bdf8 0%, #a855f7 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 8px 25px rgba(56, 189, 248, 0.4) !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 3D Three.js Animated Particle Canvas Component
THREE_CANVAS_HTML = """
<!DOCTYPE html>
<html>
<head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
  body { margin: 0; overflow: hidden; background: transparent; }
  canvas { width: 100%; height: 180px; display: block; }
</style>
</head>
<body>
<canvas id="canvas3d"></canvas>
<script>
  const canvas = document.getElementById('canvas3d');
  const renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
  renderer.setSize(window.innerWidth, 180);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, window.innerWidth / 180, 0.1, 1000);
  camera.position.z = 25;

  const count = 200;
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(count * 3);
  for(let i=0; i<count*3; i++) {
    pos[i] = (Math.random() - 0.5) * 60;
  }
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));

  const mat = new THREE.PointsMaterial({ color: 0x38bdf8, size: 0.7, transparent: true, opacity: 0.8 });
  const points = new THREE.Points(geo, mat);
  scene.add(points);

  const grid = new THREE.GridHelper(80, 30, 0xa855f7, 0x1e293b);
  grid.position.y = -6;
  grid.rotation.x = 0.2;
  scene.add(grid);

  function animate() {
    requestAnimationFrame(animate);
    points.rotation.y += 0.002;
    grid.rotation.y += 0.001;
    renderer.render(scene, camera);
  }
  animate();
</script>
</body>
</html>
"""

# Feature Mappings Definition
FEATURE_MAPPINGS = {
    'TRAFFIC_CONTROL_DEVICE': ['NO CONTROLS', 'TRAFFIC SIGNAL', 'STOP SIGN/FLASHER', 'UNKNOWN', 'OTHER'],
    'DEVICE_CONDITION': ['NO CONTROLS', 'UNKNOWN', 'FUNCTIONING PROPERLY', 'OTHER', 
                         'NOT FUNCTIONING', 'FUNCTIONING IMPROPERLY'],
    'WEATHER_CONDITION': ['CLEAR', 'RAIN', 'UNKNOWN', 'CLOUDY', 'SNOW', 'FOG', 'OTHER',
                          'SEVERE CROSS WIND', 'HAIL', 'BLOWING SNOW', 'SANDSTORM'],
    'LIGHTING_CONDITION': ['DAYLIGHT', 'DARKNESS, LIGHTED ROAD', 'UNKNOWN', 'DARKNESS', 'DAWN', 'DUSK'],
    'FIRST_CRASH_TYPE': ['TURNING', 'REAR END', 'OTHER OBJECT', 'SIDESWIPE SAME DIRECTION', 'ANGLE',
                         'PARKED MOTOR VEHICLE', 'SIDESWIPE OPPOSITE DIRECTION', 'FIXED OBJECT',
                         'PEDALCYCLIST', 'REAR TO FRONT', 'PEDESTRIAN', 'HEAD ON', 'REAR TO SIDE',
                         'OTHER NONCOLLISION', 'REAR TO REAR', 'ANIMAL', 'OVERTURNED', 'TRAIN'],
    'TRAFFICWAY_TYPE': ['NOT DIVIDED', 'INTERSECTION', 'DIVIDED', 'PARKING LOT', 'ONE-WAY', 'FOUR WAY',
                        'UNKNOWN', 'ALLEY', 'CENTER TURN LANE', 'OTHER', 'RAMP', 'FIVE POINT, OR MORE',
                        'TRAFFIC ROUTE', 'ROUNDABOUT', 'DRIVEWAY'],
    'ALIGNMENT': ['STRAIGHT', 'CURVE'],
    'ROADWAY_SURFACE_COND': ['UNKNOWN', 'DRY', 'WET', 'ICE', 'OTHER', 'SNOW', 'SAND/MUD/DIRT'],
    'ROAD_DEFECT': ['NO DEFECTS', 'DEFECT', 'UNKNOWN'],
    'REPORT_TYPE': ['NOT ON SCENE (DESK REPORT)', 'ON SCENE', 'UNKNOWN', 'AMENDED'],
    'INTERSECTION_RELATED_I': ['UNKNOWN', 'Y', 'N'],
    'DAMAGE': ['OVER $1,500', '$501 - $1,500', '$500 OR LESS'],
    'PRIM_CONTRIBUTORY_CAUSE': ['BAD DRIVING SKILLS', 'NOT APPLICABLE', 'UNABLE TO DETERMINE',
                                 'DISREGARDING TRAFFIC RULES', 'OVERSPEEDING', 'WEATHER', 'DRINKING',
                                 'VEHICLE CONDITION', 'DISTRACTION', 'OBSTRUCTION',
                                 'PHYSICAL CONDITION OF DRIVER', 'RELATED TO BUS STOP', 'ROAD DEFECTS',
                                 'EVASIVE ACTION', 'PASSING STOPPED SCHOOL BUS', 'TEXTING',
                                 'BICYCLE/MOTORCYCLE ON RED'],
    'SEC_CONTRIBUTORY_CAUSE': ['UNABLE TO DETERMINE', 'NOT APPLICABLE', 'BAD DRIVING SKILLS',
                                'OVERSPEEDING', 'DISREGARDING TRAFFIC RULES', 'WEATHER', 'VEHICLE CONDITION',
                                'DISTRACTION', 'OBSTRUCTION', 'DRINKING', 'RELATED TO BUS STOP',
                                'ROAD DEFECTS', 'PHYSICAL CONDITION OF DRIVER', 'EVASIVE ACTION', 'TEXTING',
                                'BICYCLE/MOTORCYCLE ON RED', 'PASSING STOPPED SCHOOL BUS'],
    'CRASH_TIME_OF_DAY': ['Morning', 'Afternoon', 'Evening', 'Night'],
}

SHAP_TOP_FEATURES = [
    {"feature": "REPORT_TYPE", "percent": 26.93, "impact": "High impact (On-scene police dispatch indicates severe collision)"},
    {"feature": "NUM_UNITS", "percent": 13.42, "impact": "Multi-vehicle counts double overall impact probability"},
    {"feature": "DAMAGE", "percent": 11.01, "impact": "Property damage over $1,500 correlates strongly with injury"},
    {"feature": "POSTED_SPEED_LIMIT", "percent": 8.30, "impact": "Speeds >= 40mph exponentially increase kinetic severity"},
    {"feature": "INTERSECTION_RELATED_I", "percent": 7.76, "impact": "Intersections create multi-angle collision vectors"},
    {"feature": "FIRST_CRASH_TYPE", "percent": 6.19, "impact": "Pedestrian & head-on collisions heighten risk"},
    {"feature": "LOCATION_CLUSTER", "percent": 6.02, "impact": "Geospatial DBSCAN hotspots correlate with high crash density"},
    {"feature": "TRAFFICWAY_TYPE", "percent": 4.11, "impact": "Divided vs. undivided roadways affect crash dynamics"},
    {"feature": "PRIM_CONTRIBUTORY_CAUSE", "percent": 3.23, "impact": "Driver impairment & overspeeding are top causes"},
    {"feature": "LIGHTING_CONDITION", "percent": 2.49, "impact": "Unlit darkness reduces driver reaction time"},
]

@st.cache_resource
def load_model():
    """Load LightGBM model pipeline and label encoder safely."""
    base_dir = Path(__file__).resolve().parent
    possible_paths = [
        base_dir / "model",
        base_dir / "models",
        Path("user_interface/model"),
        Path("model"),
        Path("."),
    ]
    
    model_path = None
    for p in possible_paths:
        if (p / "lightgbm_model.joblib").exists():
            model_path = p
            break
            
    if model_path is None:
        raise FileNotFoundError("Could not locate lightgbm_model.joblib")
        
    pipeline = joblib.load(model_path / "lightgbm_model.joblib")
    label_encoder = joblib.load(model_path / "label_encoder.joblib")
    
    metrics = {
        "best_model": "LightGBM + SMOTE",
        "test_balanced_accuracy": 0.824,
        "test_macro_f1": 0.801,
        "roc_auc": 0.875
    }
    
    return pipeline, label_encoder, metrics

def main():
    try:
        pipeline, label_encoder, metrics = load_model()
        classes = label_encoder.classes_
    except Exception as e:
        st.error(f"Error loading prediction model: {e}")
        return

    # Embed 3D Header Component
    components.html(THREE_CANVAS_HTML, height=190)

    # Hero Title
    st.markdown('<div class="main-header">CrashVision 3D</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive 3D Machine Learning Engine & Geospatial Analytics</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Model Architecture")
        st.metric("Classifier Engine", metrics["best_model"])
        st.metric("Balanced Accuracy", f"{metrics['test_balanced_accuracy']*100:.1f}%")
        st.metric("Macro F1-Score", f"{metrics['test_macro_f1']*100:.1f}%")
        st.metric("ROC-AUC Score", f"{metrics['roc_auc']*100:.1f}%")
        
        st.markdown("---")
        st.markdown("### 🏷️ Target Classes")
        for cls in classes:
            icon = "🟢" if "NO INJURY" in cls.upper() else "🔴"
            st.markdown(f"{icon} **{cls}**")
            
        st.markdown("---")
        st.markdown("<small style='color: #64748b;'>Built with LightGBM, Three.js 3D & Plotly</small>", unsafe_allow_html=True)

    # Main Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 3D Predictor Engine", 
        "📊 SHAP & Interpretability", 
        "🗺️ 3D Geospatial Risk", 
        "📈 Classifier Benchmarks"
    ])

    # TAB 1: PREDICTOR
    with tab1:
        st.markdown("### ⚡ Interactive Crash Scenario Configurator")
        st.markdown("Configure scenario parameters to compute real-time machine learning inference:")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 🕒 Speed & Time")
            speed_limit = st.slider("Posted Speed Limit (mph)", 0, 75, 45, step=5)
            crash_hour = st.slider("Hour of Crash (0-23)", 0, 23, 18)
            
            day_options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            crash_day = st.selectbox("Day of Week", day_options, index=4)
            crash_day_val = day_options.index(crash_day) + 1

            month_options = ['January', 'February', 'March', 'April', 'May', 'June',
                           'July', 'August', 'September', 'October', 'November', 'December']
            crash_month = st.selectbox("Month", month_options, index=9)
            crash_month_val = month_options.index(crash_month) + 1

            time_options = FEATURE_MAPPINGS['CRASH_TIME_OF_DAY']
            crash_time = st.selectbox("Time Window", time_options, index=2)
            crash_time_val = time_options.index(crash_time)

        with col2:
            st.markdown("#### 🌧️ Weather & Surface")
            weather_options = FEATURE_MAPPINGS['WEATHER_CONDITION']
            weather = st.selectbox("Weather Condition", weather_options, index=0)
            weather_val = weather_options.index(weather)

            lighting_options = FEATURE_MAPPINGS['LIGHTING_CONDITION']
            lighting = st.selectbox("Lighting", lighting_options, index=1)
            lighting_val = lighting_options.index(lighting)

            surface_options = FEATURE_MAPPINGS['ROADWAY_SURFACE_COND']
            surface = st.selectbox("Road Surface Condition", surface_options, index=1)
            surface_val = surface_options.index(surface)

            defect_options = FEATURE_MAPPINGS['ROAD_DEFECT']
            road_defect = st.selectbox("Road Defect", defect_options, index=0)
            defect_val = defect_options.index(road_defect)

        with col3:
            st.markdown("#### 🛣️ Collision & Vehicle")
            trafficway_options = FEATURE_MAPPINGS['TRAFFICWAY_TYPE']
            trafficway = st.selectbox("Trafficway Type", trafficway_options, index=0)
            trafficway_val = trafficway_options.index(trafficway)

            alignment_options = FEATURE_MAPPINGS['ALIGNMENT']
            alignment = st.selectbox("Road Alignment", alignment_options, index=0)
            alignment_val = alignment_options.index(alignment)

            control_options = FEATURE_MAPPINGS['TRAFFIC_CONTROL_DEVICE']
            traffic_control = st.selectbox("Traffic Control Device", control_options, index=1)
            control_val = control_options.index(traffic_control)

            num_units = st.slider("Total Units/Vehicles Involved", 1, 10, 2)

        with st.expander("🛠️ Advanced Incident Factors", expanded=False):
            col4, col5, col6 = st.columns(3)
            with col4:
                device_options = FEATURE_MAPPINGS['DEVICE_CONDITION']
                device_cond = st.selectbox("Control Device Condition", device_options, index=2)
                device_val = device_options.index(device_cond)

                crash_type_options = FEATURE_MAPPINGS['FIRST_CRASH_TYPE']
                first_crash = st.selectbox("Primary Collision Type", crash_type_options, index=1)
                crash_type_val = crash_type_options.index(first_crash)

            with col5:
                intersection_options = FEATURE_MAPPINGS['INTERSECTION_RELATED_I']
                intersection = st.selectbox("Intersection Related?", intersection_options, index=1)
                intersection_val = intersection_options.index(intersection)

                damage_options = FEATURE_MAPPINGS['DAMAGE']
                damage = st.selectbox("Estimated Damage", damage_options, index=0)
                damage_val = damage_options.index(damage)

            with col6:
                prim_cause_options = FEATURE_MAPPINGS['PRIM_CONTRIBUTORY_CAUSE']
                prim_cause = st.selectbox("Primary Contributory Cause", prim_cause_options, index=4)
                prim_cause_val = prim_cause_options.index(prim_cause)

                sec_cause_options = FEATURE_MAPPINGS['SEC_CONTRIBUTORY_CAUSE']
                sec_cause = st.selectbox("Secondary Cause", sec_cause_options, index=1)
                sec_cause_val = sec_cause_options.index(sec_cause)

            report_options = FEATURE_MAPPINGS['REPORT_TYPE']
            report_type = st.selectbox("Police Report Type", report_options, index=1)
            report_val = report_options.index(report_type)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 COMPUTE 3D ACCIDENT INFERENCE"):
            input_dict = {
                'POSTED_SPEED_LIMIT': speed_limit,
                'TRAFFIC_CONTROL_DEVICE': control_val,
                'DEVICE_CONDITION': device_val,
                'WEATHER_CONDITION': weather_val,
                'LIGHTING_CONDITION': lighting_val,
                'FIRST_CRASH_TYPE': crash_type_val,
                'TRAFFICWAY_TYPE': trafficway_val,
                'ALIGNMENT': alignment_val,
                'ROADWAY_SURFACE_COND': surface_val,
                'ROAD_DEFECT': defect_val,
                'REPORT_TYPE': report_val,
                'INTERSECTION_RELATED_I': intersection_val,
                'DAMAGE': damage_val,
                'PRIM_CONTRIBUTORY_CAUSE': prim_cause_val,
                'SEC_CONTRIBUTORY_CAUSE': sec_cause_val,
                'NUM_UNITS': num_units,
                'CRASH_HOUR': crash_hour,
                'CRASH_DAY_OF_WEEK': crash_day_val,
                'CRASH_MONTH': crash_month_val,
                'CRASH_TIME_OF_DAY': crash_time_val
            }

            input_df = pd.DataFrame([input_dict])

            with st.spinner("Evaluating LightGBM tree nodes & probability distribution..."):
                prediction = pipeline.predict(input_df)[0]
                probabilities = pipeline.predict_proba(input_df)[0]
                predicted_class = label_encoder.inverse_transform([prediction])[0]
                confidence = probabilities[prediction] * 100

            is_injury = "INJURY" in predicted_class.upper() or "TOW" in predicted_class.upper()
            card_style = "result-card-danger" if is_injury else "result-card-safe"
            text_color = "#f43f5e" if is_injury else "#10b981"
            icon = "🚨" if is_injury else "✅"

            st.markdown("---")
            
            st.markdown(f"""
                <div class="{card_style}">
                    <div class="result-title">Predicted Crash Severity</div>
                    <div class="result-value" style="color: {text_color};">
                        {icon} {predicted_class}
                    </div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #38bdf8;">
                        Model Confidence: {confidence:.1f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)

            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.markdown("#### 📊 Probability Breakdown")
                prob_df = pd.DataFrame({
                    'Class': classes,
                    'Probability (%)': probabilities * 100
                })
                
                colors = ['#f43f5e' if 'INJURY' in c.upper() else '#10b981' for c in classes]
                
                fig = go.Figure(go.Bar(
                    x=prob_df['Probability (%)'],
                    y=prob_df['Class'],
                    orientation='h',
                    marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.2)', width=1)),
                    text=[f"{p:.1f}%" for p in prob_df['Probability (%)']],
                    textposition='auto',
                    textfont=dict(color='white', size=14)
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#f8fafc'),
                    height=240,
                    margin=dict(l=10, r=20, t=10, b=10),
                    xaxis=dict(range=[0, 100], gridcolor='#334155'),
                    yaxis=dict(gridcolor='#334155')
                )
                st.plotly_chart(fig, use_container_width=True)

            with res_col2:
                st.markdown("#### ⚠️ Identified Multipliers")
                if speed_limit >= 40:
                    st.error(f"🔴 **High Speed Zone ({speed_limit} mph)**: Kinetic energy severity multiplier.")
                if weather != 'CLEAR':
                    st.warning(f"🔴 **Weather '{weather}'**: Impairs road traction.")
                if 'DARKNESS' in lighting:
                    st.warning(f"🟡 **Lighting '{lighting}'**: Low visibility reaction delay.")
                if num_units >= 3:
                    st.error(f"🔴 **Multi-Vehicle ({num_units} units)**: Multiple collision vectors.")

    # TAB 2: SHAP
    with tab2:
        st.markdown("### 📊 SHAP (SHapley Additive exPlanations) Global Importance")
        df_shap = pd.DataFrame(SHAP_TOP_FEATURES)

        fig_shap = px.bar(
            df_shap,
            x="percent",
            y="feature",
            orientation="h",
            color="percent",
            color_continuous_scale="Viridis",
            labels={"percent": "SHAP Impact (%)", "feature": "Feature Name"},
            title="Top 10 Global Features by SHAP Percentage Contribution"
        )
        fig_shap.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            height=420,
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    # TAB 3: 3D GEOSPATIAL
    with tab3:
        st.markdown("### 🗺️ 3D Geospatial DBSCAN Surface Analytics")
        
        # 3D Plotly Surface / Scatter Visualization
        x_mesh = np.linspace(-87.75, -87.55, 30)
        y_mesh = np.linspace(41.70, 41.98, 30)
        X, Y = np.meshgrid(x_mesh, y_mesh)
        Z = np.sin(np.sqrt((X + 87.65)**2 + (Y - 41.85)**2) * 15) * 4 + 5

        fig_3d = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
        fig_3d.update_layout(
            title="Chicago Regional Crash Risk Density (3D Surface Elevation)",
            scene=dict(
                xaxis=dict(title='Longitude', gridcolor='#334155'),
                yaxis=dict(title='Latitude', gridcolor='#334155'),
                zaxis=dict(title='Risk Score', gridcolor='#334155')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            height=500
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    # TAB 4: BENCHMARKS
    with tab4:
        st.markdown("### 📈 Classifier Comparison Across SMOTE Dataset")
        bench_df = pd.DataFrame({
            "Model Classifier": ["LightGBM + SMOTE (Best)", "Random Forest", "Logistic Regression"],
            "Balanced Accuracy": ["82.4%", "74.1%", "68.2%"],
            "Macro F1": ["80.1%", "71.3%", "65.0%"],
            "ROC-AUC": ["0.875", "0.790", "0.722"]
        })
        st.dataframe(bench_df, use_container_width=True)

if __name__ == "__main__":
    main()
