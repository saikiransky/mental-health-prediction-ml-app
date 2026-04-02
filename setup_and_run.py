import subprocess
import sys
import os

# ─────────────────────────────────────────
# STEP 1 — Check Python version
# ─────────────────────────────────────────
print("\n" + "="*55)
print("  🧠  Mental Health ML App — Setup & Launch")
print("="*55)

version = sys.version_info
print(f"\n✅  Python {version.major}.{version.minor}.{version.micro} detected")

if version.major < 3 or (version.major == 3 and version.minor < 8):
    print("❌  Python 3.8 or higher is required. Please upgrade.")
    sys.exit(1)

# ─────────────────────────────────────────
# STEP 2 — Install all packages
# ─────────────────────────────────────────
PACKAGES = [
    "streamlit",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "plotly",
    "scikit-learn",
    "openpyxl",     # Excel support (bonus)
]

print("\n📦  Installing required packages...")
print("-"*40)

for pkg in PACKAGES:
    print(f"   Installing {pkg} ...", end=" ", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg, "--quiet", "--upgrade"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("✅")
    else:
        print(f"❌  FAILED\n{result.stderr}")
        sys.exit(1)

print("\n✅  All packages installed successfully!")

# ─────────────────────────────────────────
# STEP 3 — Check dataset
# ─────────────────────────────────────────
print("\n📂  Checking for dataset...")
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "survey.csv")

if not os.path.exists(csv_path):
    print("❌  survey.csv not found in the same folder as this script.")
    print("    Please place survey.csv next to setup_and_run.py and try again.")
    sys.exit(1)
else:
    print(f"✅  survey.csv found at: {csv_path}")

# ─────────────────────────────────────────
# STEP 4 — Write the Streamlit app file
# ─────────────────────────────────────────
APP_FILENAME = "mental_health_ml_app.py"
app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), APP_FILENAME)

APP_CODE = '''"""
Mental Health Survey — ML Model Building & Deployment
Target variable: treatment (Yes / No)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)

# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health ML App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Helpers ───────────────────────────────────────────────
@st.cache_data
def load_data(path="survey.csv"):
    return pd.read_csv(path)

def clean_gender(val):
    v = str(val).strip().lower()
    if any(k in v for k in ["male","man","m","cis male","maile","mal","make","msle","mail","malr","guy"]):
        return "Male"
    elif any(k in v for k in ["female","woman","f","cis female","femail","trans-female","femake"]):
        return "Female"
    return "Other"

@st.cache_data
def preprocess(df):
    df = df.copy()
    df.drop(columns=["Timestamp","comments","state","Country"], errors="ignore", inplace=True)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df[(df["Age"] >= 15) & (df["Age"] <= 75)]
    df["Gender"] = df["Gender"].apply(clean_gender)
    df["work_interfere"] = df["work_interfere"].fillna("Don\'t know")
    df["self_employed"]  = df["self_employed"].fillna(df["self_employed"].mode()[0])
    df["treatment"]      = df["treatment"].map({"Yes": 1, "No": 0})
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object","str"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def train_all_models(X_tr, X_te, y_tr, y_te):
    models = {
        "Logistic Regression":  LogisticRegression(max_iter=1000),
        "Decision Tree":        DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM":                  SVC(probability=True, random_state=42),
        "KNN":                  KNeighborsClassifier(n_neighbors=5),
    }
    out = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        y_prob = m.predict_proba(X_te)[:, 1]
        out[name] = {
            "model":    m,
            "accuracy": accuracy_score(y_te, y_pred),
            "auc":      roc_auc_score(y_te, y_prob),
            "y_pred":   y_pred,
            "y_prob":   y_prob,
            "report":   classification_report(y_te, y_pred, output_dict=True),
        }
    return out

# ── Load & preprocess ─────────────────────────────────────
st.sidebar.title("🧠 Mental Health ML")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Data Overview", "🔍 EDA", "🤖 Model Training", "📈 Evaluation", "🎯 Predict"]
)

uploaded = st.sidebar.file_uploader("Upload a different survey CSV", type=["csv"])
try:
    raw_df = pd.read_csv(uploaded) if uploaded else load_data("survey.csv")
except FileNotFoundError:
    st.error("survey.csv not found. Please upload via the sidebar.")
    st.stop()

cleaned_df = preprocess(raw_df)
X = cleaned_df.drop("treatment", axis=1)
y = cleaned_df["treatment"]

test_pct = st.sidebar.slider("Test size (%)", 10, 40, 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_pct/100, random_state=42, stratify=y
)
scaler    = StandardScaler()
Xtr_s     = scaler.fit_transform(X_train)
Xte_s     = scaler.transform(X_test)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset info**")
st.sidebar.write(f"Rows: {len(raw_df)} | Features: {raw_df.shape[1]-1}")

# ══════════════════════════════════════════════
#  PAGE 1 — DATA OVERVIEW
# ══════════════════════════════════════════════
if page == "📊 Data Overview":
    st.title("📊 Dataset Overview")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows",        len(raw_df))
    c2.metric("Columns",     raw_df.shape[1])
    c3.metric("Needs Treatment", int((raw_df["treatment"]=="Yes").sum()))
    c4.metric("No Treatment",    int((raw_df["treatment"]=="No").sum()))

    st.subheader("Raw Data Sample")
    st.dataframe(raw_df.head(20), use_container_width=True)

    st.subheader("Missing Values & Types")
    info = pd.DataFrame({
        "Dtype":       raw_df.dtypes,
        "Non-Null":    raw_df.notnull().sum(),
        "Missing":     raw_df.isnull().sum(),
        "Missing %":   (raw_df.isnull().mean()*100).round(2),
        "Unique":      raw_df.nunique(),
    })
    st.dataframe(info, use_container_width=True)

    st.subheader("Cleaned / Encoded Data")
    st.dataframe(cleaned_df.head(20), use_container_width=True)
    st.caption(f"Shape after preprocessing: {cleaned_df.shape}")

# ══════════════════════════════════════════════
#  PAGE 2 — EDA
# ══════════════════════════════════════════════
elif page == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")

    eda = raw_df.copy()
    eda["Age"]    = pd.to_numeric(eda["Age"], errors="coerce")
    eda           = eda[(eda["Age"]>=15)&(eda["Age"]<=75)]
    eda["Gender"] = eda["Gender"].apply(clean_gender)

    st.subheader("Treatment Distribution")
    fig = px.pie(eda, names="treatment",
                 color_discrete_sequence=["#6C63FF","#FF6584"])
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age by Treatment")
        fig, ax = plt.subplots()
        sns.histplot(eda, x="Age", hue="treatment", bins=30, ax=ax,
                     palette=["#6C63FF","#FF6584"])
        st.pyplot(fig)
    with col2:
        st.subheader("Gender vs Treatment")
        fig = px.histogram(eda, x="Gender", color="treatment", barmode="group",
                           color_discrete_sequence=["#6C63FF","#FF6584"])
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Family History vs Treatment")
        fig = px.histogram(eda, x="family_history", color="treatment", barmode="group",
                           color_discrete_sequence=["#6C63FF","#FF6584"])
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        st.subheader("Work Interference vs Treatment")
        fig = px.histogram(eda, x="work_interfere", color="treatment", barmode="group",
                           color_discrete_sequence=["#6C63FF","#FF6584"])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(14,7))
    sns.heatmap(cleaned_df.corr(), cmap="coolwarm", annot=False, ax=ax)
    st.pyplot(fig)

# ══════════════════════════════════════════════
#  PAGE 3 — MODEL TRAINING
# ══════════════════════════════════════════════
elif page == "🤖 Model Training":
    st.title("🤖 Model Training")
    st.info(f"Training on **{len(X_train)}** rows | Testing on **{len(X_test)}** rows")

    if st.button("🚀 Train All 6 Models", type="primary"):
        bar = st.progress(0, text="Starting...")
        results = {}
        models_list = {
            "Logistic Regression":  LogisticRegression(max_iter=1000),
            "Decision Tree":        DecisionTreeClassifier(max_depth=5, random_state=42),
            "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
            "SVM":                  SVC(probability=True, random_state=42),
            "KNN":                  KNeighborsClassifier(n_neighbors=5),
        }
        for i, (name, m) in enumerate(models_list.items()):
            bar.progress((i)/6, text=f"Training {name}...")
            m.fit(Xtr_s, y_train)
            yp = m.predict(Xte_s)
            yb = m.predict_proba(Xte_s)[:,1]
            results[name] = {
                "model":    m,
                "accuracy": accuracy_score(y_test, yp),
                "auc":      roc_auc_score(y_test, yb),
                "y_pred":   yp,
                "y_prob":   yb,
                "report":   classification_report(y_test, yp, output_dict=True),
            }
        bar.progress(1.0, text="Done!")

        st.session_state["results"]  = results
        st.session_state["Xte_s"]    = Xte_s
        st.session_state["y_test"]   = y_test
        st.success("✅ All models trained!")

        summary = pd.DataFrame({
            "Model":    list(results.keys()),
            "Accuracy": [f"{v[\'accuracy\']*100:.2f}%" for v in results.values()],
            "ROC-AUC":  [f"{v[\'auc\']:.4f}"           for v in results.values()],
        }).sort_values("Accuracy", ascending=False).reset_index(drop=True)

        st.subheader("📋 Results Summary")
        st.dataframe(summary, use_container_width=True)

        best = max(results, key=lambda k: results[k]["accuracy"])
        st.success(f"🏆 Best Model: **{best}** — {results[best][\'accuracy\']*100:.2f}% accuracy")
    else:
        if "results" not in st.session_state:
            st.warning("Click the button above to start training.")

# ══════════════════════════════════════════════
#  PAGE 4 — EVALUATION
# ══════════════════════════════════════════════
elif page == "📈 Evaluation":
    st.title("📈 Model Evaluation")

    if "results" not in st.session_state:
        st.warning("Go to **Model Training** first.")
        st.stop()

    results = st.session_state["results"]
    y_test  = st.session_state["y_test"]
    sel     = st.selectbox("Choose a model", list(results.keys()))
    r       = results[sel]

    c1,c2,c3 = st.columns(3)
    c1.metric("Accuracy", f"{r[\'accuracy\']*100:.2f}%")
    c2.metric("ROC-AUC",  f"{r[\'auc\']:.4f}")
    c3.metric("F1 (weighted)", f"{r[\'report\'][\'weighted avg\'][\'f1-score\']:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, r["y_pred"])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No","Yes"], yticklabels=["No","Yes"], ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
    with col2:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color="#6C63FF", lw=2, label=f"AUC={r[\'auc\']:.4f}")
        ax.plot([0,1],[0,1],"k--")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
        st.pyplot(fig)

    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(r["report"]).T.round(4), use_container_width=True)

    if hasattr(r["model"], "feature_importances_"):
        st.subheader("Feature Importance")
        fi = pd.Series(r["model"].feature_importances_, index=X.columns).sort_values(ascending=False)
        fig = px.bar(fi.reset_index(), x="index", y=0,
                     labels={"index":"Feature","0":"Importance"},
                     color_discrete_sequence=["#6C63FF"])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("All Models — Accuracy vs AUC")
    comp = pd.DataFrame({
        "Model":    list(results.keys()),
        "Accuracy": [v["accuracy"] for v in results.values()],
        "AUC":      [v["auc"]      for v in results.values()],
    })
    fig = px.bar(comp.melt(id_vars="Model", var_name="Metric", value_name="Score"),
                 x="Model", y="Score", color="Metric", barmode="group",
                 color_discrete_sequence=["#6C63FF","#FF6584"])
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
#  PAGE 5 — PREDICT
# ══════════════════════════════════════════════
elif page == "🎯 Predict":
    st.title("🎯 Predict Treatment Need")

    if "results" not in st.session_state:
        st.warning("Go to **Model Training** first.")
        st.stop()

    results  = st.session_state["results"]
    best_key = max(results, key=lambda k: results[k]["accuracy"])
    st.info(f"Using best model: **{best_key}** ({results[best_key][\'accuracy\']*100:.2f}% accuracy)")

    st.markdown("### Enter Employee Details")
    c1,c2,c3 = st.columns(3)

    with c1:
        age          = st.slider("Age", 18, 70, 30)
        gender       = st.selectbox("Gender", ["Male","Female","Other"])
        self_emp     = st.selectbox("Self Employed?", ["Yes","No"])
        fam_hist     = st.selectbox("Family History?", ["Yes","No"])
        work_interf  = st.selectbox("Work Interference", ["Never","Rarely","Sometimes","Often","Don\'t know"])
        no_emp       = st.selectbox("Company Size", ["1-5","6-25","26-100","100-500","500-1000","More than 1000"])

    with c2:
        remote       = st.selectbox("Remote Work?", ["Yes","No"])
        tech         = st.selectbox("Tech Company?", ["Yes","No"])
        benefits     = st.selectbox("Mental Health Benefits?", ["Yes","No","Don\'t know"])
        care         = st.selectbox("Care Options?", ["Yes","No","Not sure"])
        wellness     = st.selectbox("Wellness Program?", ["Yes","No","Don\'t know"])
        seek         = st.selectbox("Seek Help Resources?", ["Yes","No","Don\'t know"])

    with c3:
        anon         = st.selectbox("Anonymity Protected?", ["Yes","No","Don\'t know"])
        leave        = st.selectbox("Ease of Leave?", ["Very easy","Somewhat easy","Don\'t know","Somewhat difficult","Very difficult"])
        mh_con       = st.selectbox("MH Consequence?", ["Yes","No","Maybe"])
        ph_con       = st.selectbox("PH Consequence?", ["Yes","No","Maybe"])
        cowork       = st.selectbox("Discuss w/ Coworkers?", ["Yes","No","Some of them"])
        superv       = st.selectbox("Discuss w/ Supervisor?", ["Yes","No","Some of them"])
        mh_int       = st.selectbox("MH in Interview?", ["Yes","No","Maybe"])
        ph_int       = st.selectbox("PH in Interview?", ["Yes","No","Maybe"])
        mv_phy       = st.selectbox("Mental vs Physical?", ["Yes","No","Don\'t know"])
        obs_con      = st.selectbox("Observed Consequence?", ["Yes","No"])

    if st.button("🔮 Predict Now", type="primary"):
        inp = pd.DataFrame([{
            "Age":age,"Gender":gender,"self_employed":self_emp,
            "family_history":fam_hist,"work_interfere":work_interf,
            "no_employees":no_emp,"remote_work":remote,"tech_company":tech,
            "benefits":benefits,"care_options":care,"wellness_program":wellness,
            "seek_help":seek,"anonymity":anon,"leave":leave,
            "mental_health_consequence":mh_con,"phys_health_consequence":ph_con,
            "coworkers":cowork,"supervisor":superv,"mental_health_interview":mh_int,
            "phys_health_interview":ph_int,"mental_vs_physical":mv_phy,
            "obs_consequence":obs_con,
        }])

        le = LabelEncoder()
        for col in inp.select_dtypes(include=["object"]).columns:
            inp[col] = le.fit_transform(inp[col].astype(str))
        for col in X.columns:
            if col not in inp.columns:
                inp[col] = 0
        inp = inp[X.columns]

        inp_s  = scaler.transform(inp)
        model  = results[best_key]["model"]
        pred   = model.predict(inp_s)[0]
        proba  = model.predict_proba(inp_s)[0]

        st.markdown("---")
        if pred == 1:
            st.error(f"⚠️ Employee likely **NEEDS** mental health treatment  |  Confidence: {proba[1]*100:.1f}%")
        else:
            st.success(f"✅ Employee likely does **NOT** need treatment  |  Confidence: {proba[0]*100:.1f}%")

        prob_df = pd.DataFrame({
            "Outcome":["No Treatment","Needs Treatment"],
            "Probability":[proba[0], proba[1]]
        })
        fig = px.bar(prob_df, x="Outcome", y="Probability", color="Outcome",
                     text_auto=".2%", color_discrete_sequence=["#6C63FF","#FF6584"])
        fig.update_layout(yaxis_range=[0,1], showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("⚕️ For informational purposes only — not a clinical diagnosis.")
'''

print(f"\n📝  Writing app file: {app_path}")
with open(app_path, "w", encoding="utf-8") as f:
    f.write(APP_CODE)
print("✅  App file written!")

# ─────────────────────────────────────────
# STEP 5 — Launch Streamlit
# ─────────────────────────────────────────
print("\n🚀  Launching Streamlit app...")
print("    Your browser should open automatically.")
print("    If not, open: http://localhost:8501")
print("\n    Press Ctrl+C to stop the app.\n")
print("="*55 + "\n")

subprocess.run(
    [sys.executable, "-m", "streamlit", "run", app_path,
     "--server.headless", "false",
     "--browser.gatherUsageStats", "false"],
    cwd=os.path.dirname(os.path.abspath(__file__))
)
