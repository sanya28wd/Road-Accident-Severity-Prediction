import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
from pathlib import Path

st.set_page_config(
    page_title="Road Accident Severity Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Refined Modern Enterprise Design System CSS with Glow & Animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=Inter:wght@400;500;600;700&display=swap');
    
    * { 
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #060913;
        background-image: 
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.12) 0px, transparent 50%);
        color: #f8fafc;
    }
    
    /* Typography & Hierarchy */
    h1, h2, h3, h4 {
        letter-spacing: -0.02em;
    }
    
    .stApp p, .stApp label, .stApp span, .stApp div, .stMarkdown {
        color: #e2e8f0 !important;
    }
    
    /* Header Container */
    .app-header {
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .app-badge {
        display: inline-block;
        padding: 4px 12px;
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.2), rgba(6, 182, 212, 0.2));
        border: 1px solid rgba(6, 182, 212, 0.4);
        border-radius: 9999px;
        color: #06b6d4;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }

    .app-title {
        font-size: 2.6rem;
        font-weight: 900;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
    }
    
    .app-subtitle {
        color: #94a3b8 !important;
        font-size: 1.05rem;
    }

    /* Metric Cards with Glowing Border */
    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 24px;
        transition: all 0.3s ease;
    }

    [data-testid="stMetric"]:hover {
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 10px 25px rgba(0,0,0,0.3), 0 0 15px rgba(99, 102, 241, 0.2);
    }

    [data-testid="stMetricValue"] {
        font-size: 1.9rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
    }

    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* Section Panels */
    .clean-panel {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 26px;
        margin-bottom: 20px;
        backdrop-filter: blur(16px);
    }

    /* Verdict Banners with Neon Glow */
    .verdict-box-danger {
        background: linear-gradient(180deg, rgba(244, 63, 94, 0.15) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 2px solid #f43f5e;
        border-radius: 22px;
        padding: 28px;
        text-align: center;
        margin-bottom: 22px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.5), 0 0 30px rgba(244, 63, 94, 0.3);
    }

    .verdict-box-safe {
        background: linear-gradient(180deg, rgba(16, 185, 129, 0.15) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 2px solid #10b981;
        border-radius: 22px;
        padding: 28px;
        text-align: center;
        margin-bottom: 22px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.5), 0 0 30px rgba(16, 185, 129, 0.3);
    }

    .verdict-title {
        font-size: 1.65rem;
        font-weight: 900;
        margin: 8px 0;
    }

    /* Primary Action Button with Gradient */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #06b6d4 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 16px 32px !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        letter-spacing: 0.05em !important;
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 25px rgba(6, 182, 212, 0.5) !important;
    }

    /* Inputs & Selectboxes */
    div[data-baseweb="select"] > div {
        background-color: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
    }

    div[data-baseweb="select"] span {
        color: #ffffff !important;
    }

    /* Clean Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 6px;
        margin-bottom: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 10px !important;
        color: #94a3b8 !important;
        border: none !important;
        padding: 10px 22px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.35) !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Feature Mappings
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
    {"feature": "REPORT_TYPE (On-Scene Dispatch)", "percent": 26.93},
    {"feature": "NUM_UNITS (Vehicles Involved)", "percent": 13.42},
    {"feature": "DAMAGE (> $1,500 Severity)", "percent": 11.01},
    {"feature": "POSTED_SPEED_LIMIT (mph)", "percent": 8.30},
    {"feature": "INTERSECTION_RELATED_I", "percent": 7.76},
    {"feature": "FIRST_CRASH_TYPE", "percent": 6.19},
    {"feature": "LOCATION_CLUSTER (Hotspot)", "percent": 6.02},
    {"feature": "TRAFFICWAY_TYPE", "percent": 4.11},
    {"feature": "PRIM_CONTRIBUTORY_CAUSE", "percent": 3.23},
    {"feature": "LIGHTING_CONDITION", "percent": 2.49},
]

@st.cache_resource
def load_model():
    """Load model files from model directory"""
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
        raise FileNotFoundError("Could not find lightgbm_model.joblib")
        
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
        st.error(f"Error loading model: {e}")
        return

    # Header Bar
    st.markdown("""
        <div class="app-header">
            <span class="app-badge">⚡ Advanced ML & Geospatial Analytics</span>
            <h1 class="app-title">Road Accident Severity Predictor</h1>
            <p class="app-subtitle">Machine learning inference and geospatial intelligence for real-time traffic injury risk assessment.</p>
        </div>
    """, unsafe_allow_html=True)

    # Top KPI Metrics Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Balanced Accuracy", "82.4%", "+14.2% vs Baseline")
    with kpi2:
        st.metric("Macro F1-Score", "0.801", "SMOTE Balanced")
    with kpi3:
        st.metric("ROC-AUC Score", "0.875", "Validation Set")
    with kpi4:
        st.metric("Inference Engine", "LightGBM", "Real-Time Inference")

    st.markdown("<br>", unsafe_allow_html=True)

    # Workspace Tabs
    tab_studio, tab_shap, tab_geo, tab_model = st.tabs([
        "Prediction Studio", 
        "SHAP Explainability", 
        "Geospatial Intelligence", 
        "Model Specifications"
    ])

    # 1. PREDICTION STUDIO
    with tab_studio:
        col_left, col_right = st.columns([1.3, 1.0], gap="large")

        with col_left:
            # Panel 1: Road & Speed
            with st.container():
                st.markdown("#### Roadway & Speed Dynamics")
                speed_limit = st.slider("Posted Speed Limit (mph)", 15, 75, 45, step=5)
                
                sub_c1, sub_c2 = st.columns(2)
                with sub_c1:
                    trafficway_options = FEATURE_MAPPINGS['TRAFFICWAY_TYPE']
                    trafficway = st.selectbox("Trafficway Type", trafficway_options, index=0)
                    trafficway_val = trafficway_options.index(trafficway)
                with sub_c2:
                    intersection_options = ['Y', 'N', 'UNKNOWN']
                    intersection = st.selectbox("Intersection Related?", intersection_options, index=0)
                    intersection_val = intersection_options.index(intersection)

            st.markdown("---")

            # Panel 2: Environmental
            with st.container():
                st.markdown("#### Environmental & Lighting Factors")
                sub_e1, sub_e2 = st.columns(2)
                with sub_e1:
                    weather_options = FEATURE_MAPPINGS['WEATHER_CONDITION']
                    weather = st.selectbox("Weather Condition", weather_options, index=0)
                    weather_val = weather_options.index(weather)
                    
                    crash_hour = st.slider("Hour of Incident", 0, 23, 18)
                with sub_e2:
                    lighting_options = FEATURE_MAPPINGS['LIGHTING_CONDITION']
                    lighting = st.selectbox("Lighting Environment", lighting_options, index=1)
                    lighting_val = lighting_options.index(lighting)
                    
                    surface_options = FEATURE_MAPPINGS['ROADWAY_SURFACE_COND']
                    surface = st.selectbox("Road Surface Condition", surface_options, index=1)
                    surface_val = surface_options.index(surface)

            st.markdown("---")

            # Panel 3: Incident Details
            with st.container():
                st.markdown("#### Incident Characteristics")
                sub_i1, sub_i2 = st.columns(2)
                with sub_i1:
                    crash_type_options = FEATURE_MAPPINGS['FIRST_CRASH_TYPE']
                    first_crash = st.selectbox("Primary Collision Type", crash_type_options, index=1)
                    crash_type_val = crash_type_options.index(first_crash)

                    prim_cause_options = FEATURE_MAPPINGS['PRIM_CONTRIBUTORY_CAUSE']
                    prim_cause = st.selectbox("Contributing Cause", prim_cause_options, index=4)
                    prim_cause_val = prim_cause_options.index(prim_cause)
                with sub_i2:
                    damage_options = FEATURE_MAPPINGS['DAMAGE']
                    damage = st.selectbox("Estimated Damage Value", damage_options, index=0)
                    damage_val = damage_options.index(damage)

                    num_units = st.slider("Total Vehicles Involved", 1, 8, 2)

                report_options = FEATURE_MAPPINGS['REPORT_TYPE']
                report_type = st.selectbox("Police Report Status", report_options, index=1)
                report_val = report_options.index(report_type)

            compute_btn = st.button("RUN SEVERITY INFERENCE", use_container_width=True)

        with col_right:
            # Model Inference Calculation
            time_options = FEATURE_MAPPINGS['CRASH_TIME_OF_DAY']
            time_val = 2
            day_val = 5
            month_val = 10
            align_val = 0
            defect_val = 0
            device_val = 2
            control_val = 1
            sec_cause_val = 1

            input_dict = {
                'POSTED_SPEED_LIMIT': speed_limit,
                'TRAFFIC_CONTROL_DEVICE': control_val,
                'DEVICE_CONDITION': device_val,
                'WEATHER_CONDITION': weather_val,
                'LIGHTING_CONDITION': lighting_val,
                'FIRST_CRASH_TYPE': crash_type_val,
                'TRAFFICWAY_TYPE': trafficway_val,
                'ALIGNMENT': align_val,
                'ROADWAY_SURFACE_COND': surface_val,
                'ROAD_DEFECT': defect_val,
                'REPORT_TYPE': report_val,
                'INTERSECTION_RELATED_I': intersection_val,
                'DAMAGE': damage_val,
                'PRIM_CONTRIBUTORY_CAUSE': prim_cause_val,
                'SEC_CONTRIBUTORY_CAUSE': sec_cause_val,
                'NUM_UNITS': num_units,
                'CRASH_HOUR': crash_hour,
                'CRASH_DAY_OF_WEEK': day_val,
                'CRASH_MONTH': month_val,
                'CRASH_TIME_OF_DAY': time_val
            }

            input_df = pd.DataFrame([input_dict])
            prediction = pipeline.predict(input_df)[0]
            probabilities = pipeline.predict_proba(input_df)[0]
            predicted_class = label_encoder.inverse_transform([prediction])[0]
            confidence = probabilities[prediction] * 100

            is_injury = "INJURY" in predicted_class.upper() or "TOW" in predicted_class.upper()
            box_style = "verdict-box-danger" if is_injury else "verdict-box-safe"
            text_color = "#f43f5e" if is_injury else "#10b981"
            verdict_badge = "CRITICAL SEVERITY INCIDENT" if is_injury else "LOW SEVERITY / PROPERTY DAMAGE"

            st.markdown(f"""
                <div class="{box_style}">
                    <div style="font-size: 0.82rem; font-weight: 800; letter-spacing: 0.08em; color: {text_color}; text-transform: uppercase;">
                        ● {verdict_badge}
                    </div>
                    <div class="verdict-title" style="color: {text_color};">
                        {predicted_class}
                    </div>
                    <div style="font-size: 1.15rem; color: #94a3b8; font-weight: 600;">
                        Model Confidence: <strong style="color: #ffffff;">{confidence:.1f}%</strong>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Probability Distribution Chart
            st.markdown("##### Probability Breakdown")
            prob_df = pd.DataFrame({
                'Class': classes,
                'Probability (%)': probabilities * 100
            })
            colors = ['#f43f5e' if 'INJURY' in c.upper() else '#10b981' for c in classes]

            fig = go.Figure(go.Bar(
                x=prob_df['Probability (%)'],
                y=prob_df['Class'],
                orientation='h',
                marker=dict(color=colors),
                text=[f"{p:.1f}%" for p in prob_df['Probability (%)']],
                textposition='auto',
                textfont=dict(color='white', size=13)
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8', family='Plus Jakarta Sans'),
                height=180,
                margin=dict(l=10, r=20, t=10, b=10),
                xaxis=dict(range=[0, 100], gridcolor='rgba(255,255,255,0.06)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.06)')
            )
            st.plotly_chart(fig, use_container_width=True)

            # Key Multipliers
            st.markdown("##### Key Factor Highlights")
            if speed_limit >= 45:
                st.markdown("⚡ **Speed Limit**: ≥45 mph increases collision kinetic dissipation.")
            if damage == 'OVER $1,500':
                st.markdown("💥 **Structural Damage**: Heavy damage correlates with occupant injury.")
            if num_units >= 3:
                st.markdown("🚗 **Multi-Vehicle**: Multiple impacts heighten secondary collision risks.")
            if weather != 'CLEAR':
                st.markdown(f"🌧️ **Weather Impact**: Reduced friction from '{weather}'.")

    # 2. SHAP EXPLAINABILITY
    with tab_shap:
        st.markdown("### Global Feature Contribution (SHAP TreeExplainer)")
        st.markdown("Game-theoretic Shapley values calculated on test splits show the top drivers of prediction decisions:")

        df_shap = pd.DataFrame(SHAP_TOP_FEATURES)
        fig_shap = px.bar(
            df_shap,
            x="percent",
            y="feature",
            orientation="h",
            labels={"percent": "Mean SHAP Impact (%)", "feature": "Feature"},
            color="percent",
            color_continuous_scale="Viridis"
        )
        fig_shap.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', family='Plus Jakarta Sans'),
            height=420,
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    # 3. GEOSPATIAL INTELLIGENCE
    with tab_geo:
        st.markdown("### Chicago Spatial Hotspots & Severity Rankings")
        st.markdown("Evaluated using DBSCAN clustering with Haversine metrics and severity weighting formula:")
        
        geo_table = pd.DataFrame({
            "Cluster ID": ["#C-01", "#C-02", "#C-03", "#C-04", "#C-05"],
            "Postal Code / Corridor": ["60607 (Loop / I-90 Junction)", "60616 (Near South Side)", "60618 (Avondale / Elston)", "60647 (Logan Square)", "60630 (Jefferson Park)"],
            "Reported Volume": ["1,420 crashes", "1,150 crashes", "820 crashes", "740 crashes", "410 crashes"],
            "Severity Index": ["8.92 / 10.0", "8.45 / 10.0", "6.30 / 10.0", "5.85 / 10.0", "3.20 / 10.0"],
            "Priority Tier": ["Critical Priority", "Critical Priority", "Elevated", "Elevated", "Standard Baseline"]
        })
        st.dataframe(geo_table, use_container_width=True)

    # 4. MODEL SPECIFICATIONS
    with tab_model:
        st.markdown("### Candidate Model Performance Comparison")
        model_table = pd.DataFrame({
            "Algorithm": ["LightGBM Classifier (Production)", "Random Forest (100 Trees)", "Logistic Regression"],
            "Sampling": ["SMOTE (1:1 Ratio)", "Random Undersampling", "Class Weight Balanced"],
            "Balanced Accuracy": ["82.4%", "74.1%", "68.2%"],
            "Macro F1": ["0.801", "0.713", "0.650"],
            "ROC-AUC": ["0.875", "0.790", "0.722"]
        })
        st.dataframe(model_table, use_container_width=True)

if __name__ == "__main__":
    main()
