# Road Accident Severity Prediction 🚗⚠️

This repository contains our end-to-end **Road Accident Severity Prediction** project built as part of the Foundations of Data Science coursework. The goal of this project is to build a reliable machine learning pipeline that predicts the severity of a traffic crash and identifies high-risk areas using geospatial intelligence.

**Key Features:**
- 🤖 **Advanced ML Pipeline**: LightGBM classifier with SMOTE for imbalanced data
- 📊 **Model Explainability**: SHAP and LIME interpretability analysis
- 🗺️ **Geospatial Intelligence**: Hotspot detection and risk scoring
- 🎨 **Interactive UI**: Streamlit web application for predictions
- 📈 **Comprehensive Reporting**: Visual dashboards and performance metrics

---

## Dataset

We use the **Chicago Crash Dataset**, which contains detailed reports of road accidents, including severity, environmental conditions, vehicle details, and locational information.

 **Dataset link:** *[https://www.kaggle.com/datasets/nathaniellybrand/chicago-car-crash-dataset]*

---

## Project Workflow

### **Data Preprocessing**
- Data cleaning and validation
- Handling missing values and outliers
- Feature engineering and encoding
- Train-test split with stratification
- **Output:** Clean, model-ready dataset

### **Model Training**
- **Algorithm:** LightGBM with SMOTE
- **Approach:** Binary classification (Injury vs. No Injury)
- **Pipeline:** Preprocessing + SMOTE + LightGBM classifier
- **Optimization:** Stratified K-fold cross-validation
- **Class Imbalance:** Addressed using SMOTE oversampling and balanced class weights

### **Model Explainability**
Understanding model predictions through advanced interpretability techniques:

#### **SHAP (SHapley Additive exPlanations)**
SHAP provides comprehensive model interpretability through game theory-based feature importance:
- **Global Feature Importance** (`shap_top20_bar.png`): Displays the top 20 features ranked by their average impact on model predictions
- **Waterfall Plots** (`shap_force_sample_*.html`): Interactive HTML visualizations showing how features contribute to individual predictions
  - Sample predictions (10, 100, 250 records) demonstrate feature contribution patterns
  - Visualize positive vs. negative impact of each feature on accident severity prediction
- **Feature Interaction Analysis**: Identifies which features work together to influence predictions
- **SHAP Force Plots**: Interactive decision transparency showing feature values and their directional impact

**Generated Outputs:**
- `modelling/reports/explainability_outputs/shap_top20_bar.png` - Feature importance bar chart
- `modelling/reports/explainability_outputs/shap_force_sample_*.html` - Interactive prediction explanations
- `modelling/reports/explainability_outputs/shap_top20_percent.csv` - Feature contribution percentages
- `modelling/reports/explainability_outputs/explainability_report.html` - Comprehensive explainability report

#### **LIME (Local Interpretable Model-agnostic Explanations)**
LIME provides local approximations for instance-specific predictions:
- Local approximations of model behavior around specific predictions
- Instance-specific feature contributions explaining why a particular crash was classified as severe or not
- Model-agnostic approach ensures interpretability regardless of the ML algorithm used
- Human-readable explanations for individual accident predictions

### **Geospatial Analysis**
Comprehensive spatial risk assessment pipeline:

**a) Crash Point Geometry**
- Validation of latitude/longitude coordinates
- Conversion to GeoDataFrame
- Export as GeoJSON for spatial operations

**b) Reverse Geocoding**
- Spatial join with postal area (POA) polygons
- Assignment of POA codes and names to crash points
- Geographical aggregation of crash data

**c) Hotspot Detection (DBSCAN)**
- Haversine-distance based clustering
- Identification of spatial crash clusters
- Noise point detection (outlier crashes)
- Cluster labeling and characterization

**d) Risk Score Computation**
Severity-weighted risk scoring system:
```
Risk Score = (3 × Killed + 2 × Serious + 1 × Moderate + 0.5 × Minor) / Total Crashes
```
Generated outputs:
- ZIP/Postcode risk tables
- Cluster risk tables
- Area-level risk rankings

**e) Interactive Visualizations**

**Choropleth Maps:**
- Color-coded risk intensity by region
- Smooth Viridis color gradient
- Interactive hover information
- Clear severity shading

**Hotspot Maps:**
- Multi-layer visualization with:
  - Raw crash points
  - DBSCAN clusters (size and color by severity)
  - Heatmap overlay
  - Postcode boundaries
  - Layer toggle controls

**Interactive Dashboard:**
- Real-time metric selection
- Configurable crash count filters
- Dynamic color scale adjustment
- Live Mapbox choropleth updates
- Comprehensive crash risk explorer

### **User Interface**
**Streamlit Web Application** (`user_interface/app.py`):
- 🎯 Interactive prediction interface for real-time accident severity assessment
- 📊 Risk factor visualization and feature importance charts
- 🔍 Model prediction explanations with confidence scores
- 🚀 Easy-to-use dashboard for stakeholders and data scientists
- Pre-trained LightGBM model integrated with feature preprocessing pipeline

### **Project Structure**

```
Road-Accident-Severity-Prediction/
├── data/                          # Dataset storage
│   └── processed/                 # Cleaned and labeled data
├── data_preprocessing/            # Data cleaning and feature engineering
├── modelling/                     # ML model training and evaluation
│   ├── modelling.py               # Model training pipeline
│   ├── explainability.py          # SHAP and LIME analysis
│   ├── models/                    # Trained model artifacts
│   ├── reports/                   # Performance visualizations
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance_LightGBM.png
│   │   ├── cross validation metrics comparison.png
│   │   └── explainability_outputs/
│   │       ├── shap_top20_bar.png
│   │       ├── shap_force_sample_*.html
│   │       └── shap_top20_percent.csv
├── geo_spatial_risk/              # Geospatial analysis and visualizations
│   └── Chicago/                   # Location-specific analysis
├── user_interface/                # Streamlit web application
│   ├── app.py
│   ├── prepare_data.py
│   ├── model/                     # Trained models and encoders
│   └── processed_data/            # Preprocessed features
└── README.md

```

### **Installation & Setup**

#### **Requirements:**
- Python 3.8+
- Libraries: pandas, scikit-learn, lightgbm, shap, lime, streamlit, folium, geopandas

#### **Installation:**
```bash
# Clone repository
git clone https://github.com/sanya28wd/Road-Accident-Severity-Prediction.git
cd Road-Accident-Severity-Prediction

# Install dependencies
pip install -r user_interface/requirements.txt

# Run the Streamlit application
streamlit run user_interface/app.py
```

---

##  Model Performance & Visualization

### **Performance Metrics**

| Metric | Value |
|--------|-------|
| **Balanced Accuracy** | 79.0% |
| **Macro F1-Score** | 80.0% |

### **Model Evaluation Reports**

**Confusion Matrix:** (`modelling/reports/confusion_matrix.png`)
- Visual representation of true positives, false positives, true negatives, and false negatives
- Helps assess model's classification performance across both injury and non-injury categories

**Feature Importance:** (`modelling/reports/feature_importance_LightGBM.png`)
- Top predictive features driving accident severity predictions
- Identifies which traffic, environmental, and vehicle factors matter most

**Cross-Validation Metrics:** (`modelling/reports/cross validation metrics comparison.png`)
- Stratified K-fold cross-validation performance across folds
- Demonstrates model stability and generalization capability

---
##  Contributors

This project was developed as part of a Foundations of Data Science group project by passionate data scientists and machine learning engineers.

| Name | GitHub | Contribution |
|------|--------|--------------|
| **Shreiya** | [@Shreiya-Muthuvelan](https://github.com/Shreiya-Muthuvelan) | Data preprocessing and Streamlit UI development |
| **Sanya** | [@sanya28wd](https://github.com/sanya28wd) | Modelling using Logistic Regression, Extra Trees, XGBoost, LightGBM |
| **Aarushi** | [@aarushi4-ux](https://github.com/aarushi4-ux) | Model explainability (SHAP/LIME), documentation |
| **Chirudeva** | [@Tactical-Camell](https://github.com/Tactical-Camell) | Geospatial analysis, hotspot detection, risk scoring |

---

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Support & Feedback

For questions, issues, or contributions, please open an issue or submit a pull request. We welcome feedback and contributions from the community!

---

**Last Updated:** August 2024
**Status:** ✅ Production Ready
