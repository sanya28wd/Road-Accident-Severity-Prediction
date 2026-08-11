
import os 
import json 
from pathlib import Path

import joblib 

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt  
import seaborn as sns 

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import lightgbm as lgb

PROJECT_ROOT = Path("/Users/sanyawadhawan/Desktop/road-accident-severity")

DATA_PATH = Path("/Users/sanyawadhawan/Desktop/data_labeled_new.csv")
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Dataset shape (rows, columns):", df.shape)
print("First 5 rows:")
print(df.head(5))

TARGET = "CRASH_TYPE"

assert TARGET in df.columns, f"Target column '{TARGET}' not found in data!"

X = df.drop(columns=[TARGET])

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(df[TARGET]) 
joblib.dump(le, MODELS_DIR / "label_encoder.joblib")

print("\nFeature columns (X):", list(X.columns))
print(pd.Series(y).value_counts(normalize=True).sort_index())

numeric_features = X.columns.tolist()
print("\nNumber of numeric features:", len(numeric_features))


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,       
    stratify=y,   
    random_state=RANDOM_STATE,
)

print("\nTrain size:", X_train.shape, "Test size:", X_test.shape)
print("Class distribution in full data:")
print(pd.Series(y).value_counts(normalize=True).sort_index())


numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),  
        ("scaler", StandardScaler()),                 
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
    ],
    remainder="drop", 
)


pipe_logreg = Pipeline(
    steps=[
        ("preprocess", preprocess),
        (
            "clf",
            LogisticRegression(
                max_iter=2000,
                multi_class="auto",
                solver="saga",
                class_weight="balanced", 
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)


pipe_extratrees = Pipeline(
    steps=[
        ("preprocess", preprocess),
        (
            "clf",
            ExtraTreesClassifier(
                n_estimators=400,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,  
            ),
        ),
    ]
)


pipe_xgb = Pipeline(
    steps=[
        ("preprocess", preprocess),
        (
            "clf",
            xgb.XGBClassifier(
                objective="binary:logistic",  
                eval_metric="logloss",        
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)

pipe_lgbm = Pipeline(
    steps=[
        ("preprocess", preprocess),
        (
            "clf",
            lgb.LGBMClassifier(
                objective="binary",            
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)

smote = SMOTE(random_state=RANDOM_STATE)

pipe_logreg_smote = ImbPipeline(
    steps=[
        ("preprocess", preprocess),
        ("smote", smote),  
        (
            "clf",
            LogisticRegression(
                max_iter=2000,
                multi_class="auto",
                solver="saga",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)


scoring = {
    "macro_f1": make_scorer(f1_score, average="macro"),   
    "bal_acc": make_scorer(balanced_accuracy_score), 
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)


def evaluate_with_cv(model, name):  

    print(f"\nRunning CV for model: {name}")
    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1, 
        return_estimator=False,
    )

    summary = {
        "model": name,
        "macro_f1_mean": float(np.mean(cv_results["test_macro_f1"])),
        "macro_f1_std": float(np.std(cv_results["test_macro_f1"])),
        "bal_acc_mean": float(np.mean(cv_results["test_bal_acc"])),
        "bal_acc_std": float(np.std(cv_results["test_bal_acc"])),
    }

    print("CV results for", name, ":", summary)
    return summary

models_to_try = [
    (pipe_logreg, "LogisticRegression"),
    (pipe_extratrees, "ExtraTrees"),
    (pipe_xgb, "XGBoost"),
    (pipe_lgbm, "LightGBM"),
    (pipe_logreg_smote, "LogReg+SMOTE"),
]

all_cv_results = []
for mdl, name in models_to_try:
    try:
        result = evaluate_with_cv(mdl, name)
        all_cv_results.append(result)
    except Exception as e:
        print(f"Error while evaluating {name}: {e}")
        all_cv_results.append({"model": name, "error": str(e)})

cv_results_df = pd.DataFrame(all_cv_results) 
print("\nCross-validation summary:")
print(cv_results_df)

cv_plot = cv_results_df.copy()
cv_plot = cv_plot[["model", "macro_f1_mean", "bal_acc_mean"]].set_index("model")

cv_plot.plot(kind="bar", figsize=(10,6), rot=45, edgecolor="black")
plt.title("Cross-Validation Metrics Comparison")
plt.ylabel("Score")
plt.ylim(0,1)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


valid_results = [r for r in all_cv_results if "error" not in r]

if len(valid_results) > 0:
    cv_results_valid_df = pd.DataFrame(valid_results)
    best_row = cv_results_valid_df.sort_values(
        "macro_f1_mean", ascending=False
    ).iloc[0]
    best_model_name = best_row["model"]
else:
    best_model_name = "ExtraTrees"

print("\nBest model based on CV macro F1:", best_model_name)

model_map = {
    "LogisticRegression": pipe_logreg,
    "ExtraTrees": pipe_extratrees,
    "XGBoost": pipe_xgb,
    "LightGBM": pipe_lgbm,
    "LogReg+SMOTE": pipe_logreg_smote,
}

best_model = model_map[best_model_name]

print("\nFitting the best model on full training data...")
best_model.fit(X_train, y_train)  

if best_model_name in ["ExtraTrees", "XGBoost", "LightGBM"]:
    model_feat = best_model.named_steps['clf']
    importances = model_feat.feature_importances_
    features = X.columns
    feat_df = pd.DataFrame({"feature": features, "importance": importances})
    feat_df = feat_df.sort_values(by="importance", ascending=False).head(20)  # top 20

    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x="importance", y="feature", data=feat_df, palette="viridis", ax=ax)
    ax.set_title(f"Top 20 Feature Importances - {best_model_name}")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / f"feature_importance_{best_model_name}.png", dpi=150)
    plt.show()

y_pred = best_model.predict(X_test)

try:
    y_proba = best_model.predict_proba(X_test)
except Exception:
    y_proba = None

test_bal_acc = balanced_accuracy_score(y_test, y_pred)
test_macro_f1 = f1_score(y_test, y_pred, average="macro")

print("\n=== Test set performance ===")
print("Best model:", best_model_name)
print("Test Balanced Accuracy:", test_bal_acc)
print("Test Macro F1:", test_macro_f1)
print("\nClassification report:\n")
print(classification_report(y_test, y_pred,target_names=le.classes_))


cm = confusion_matrix(y_test, y_pred)
class_labels = le.classes_ 
cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax)
ax.set_title(f"Confusion Matrix - {best_model_name}")
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")
plt.tight_layout()
fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
plt.show()

model_path = MODELS_DIR / "best_pipeline.joblib"
joblib.dump(best_model, model_path)
print("\nSaved best model to:", model_path)

cv_results_path = REPORTS_DIR / "cv_results.csv"
cv_results_df.to_csv(cv_results_path, index=False)
print("Saved CV results to:", cv_results_path)

metrics = {
    "best_model": best_model_name,
    "test_balanced_accuracy": test_bal_acc,
    "test_macro_f1": test_macro_f1,
    "classes": le.classes_.tolist(),
}
metrics_path = REPORTS_DIR / "metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved test metrics to:", metrics_path)


if y_proba is not None:
    proba_df = pd.DataFrame(y_proba, index=X_test.index)
    proba_path = REPORTS_DIR / "test_proba.parquet"
    proba_df.to_parquet(proba_path)
    print("Saved test probabilities to:", proba_path)

print("\nDone!")

