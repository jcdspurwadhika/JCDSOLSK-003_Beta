import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉", layout="wide")

# ----------------------------
# Paths & artifact bootstrap
# ----------------------------
BASE_DIR = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
ZIP_FALLBACK = BASE_DIR / "telco_churn_artifacts.zip"

REQUIRED_FILES = ["best_model.joblib", "scaler.joblib", "feature_columns.joblib"]

def ensure_artifacts():
    """Ensure artifacts exist in ./artifacts.
    If not, try extracting from telco_churn_artifacts.zip in the project root.
    """
    ARTIFACT_DIR.mkdir(exist_ok=True)
    missing = [f for f in REQUIRED_FILES if not (ARTIFACT_DIR / f).exists()]
    if not missing:
        return

    if ZIP_FALLBACK.exists():
        with zipfile.ZipFile(ZIP_FALLBACK, "r") as z:
            z.extractall(ARTIFACT_DIR)

    missing = [f for f in REQUIRED_FILES if not (ARTIFACT_DIR / f).exists()]
    if missing:
        st.error(
            "Artifacts belum lengkap. Pastikan folder ./artifacts berisi: "
            + ", ".join(REQUIRED_FILES)
            + " (atau taruh telco_churn_artifacts.zip di root project)."
        )
        st.stop()

@st.cache_resource
def load_artifacts():
    ensure_artifacts()
    try:
        model = joblib.load(ARTIFACT_DIR / "best_model.joblib")
        scaler = joblib.load(ARTIFACT_DIR / "scaler.joblib")
        feature_cols = joblib.load(ARTIFACT_DIR / "feature_columns.joblib")
    except Exception as e:
        st.error(
            "Gagal load artifact. Error seperti `numpy.core.multiarray failed to import` biasanya karena "
            "bentrok versi numpy/scikit-learn. Solusi: buat venv baru dan install versi yang dipin "
            "(lihat requirements.txt).\n\n"
            f"Detail error: {e}"
        )
        st.stop()

    return model, scaler, feature_cols

model, scaler, FEATURE_COLS = load_artifacts()

# ----------------------------
# Schema (raw input)
# ----------------------------
RAW_COLUMNS = [
    "gender",
    "SeniorCitizen",     # 0/1
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORIES = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

def to_model_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Convert 1-row raw input into encoded feature space using get_dummies(drop_first=True),
    then align to FEATURE_COLS order.
    """
    raw_df = raw_df.copy()
    raw_df["SeniorCitizen"] = raw_df["SeniorCitizen"].astype(int)
    raw_df["tenure"] = raw_df["tenure"].astype(float)
    raw_df["MonthlyCharges"] = raw_df["MonthlyCharges"].astype(float)
    raw_df["TotalCharges"] = raw_df["TotalCharges"].astype(float)

    X = pd.get_dummies(raw_df, drop_first=True)
    X = X.reindex(columns=FEATURE_COLS, fill_value=0)
    return X

def predict_one(raw_input: dict):
    raw_df = pd.DataFrame([raw_input], columns=RAW_COLUMNS)
    X = to_model_features(raw_df)
    X_scaled = scaler.transform(X.values)

    # probability for class 1 (churn)
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X_scaled)[0, 1])
    else:
        scores = model.decision_function(X_scaled)
        proba = float(1 / (1 + np.exp(-scores[0])))

    pred = int(proba >= 0.5)  # fixed threshold
    return pred, proba, X

def top_coefficients(n=12):
    """Top coefficients for Logistic Regression (if available)."""
    if not hasattr(model, "coef_"):
        return None
    coef = model.coef_.ravel()
    dfc = pd.DataFrame({"feature": FEATURE_COLS, "coef": coef})
    dfc["abs_coef"] = dfc["coef"].abs()
    dfc = dfc.sort_values("abs_coef", ascending=False).head(n)
    dfc["impact"] = np.where(dfc["coef"] > 0, "↑ meningkatkan churn", "↓ menurunkan churn")
    return dfc[["feature", "coef", "impact"]]

# ----------------------------
# UI
# ----------------------------
st.title("📉 Telco Customer Churn Predictor")
st.caption("Versi sederhana: tanpa setting threshold, fokus prediksi 1 customer + penjelasan faktor churn.")

with st.expander("ℹ️ Kolom yang berdampak ke churn (ringkas)"):
    st.markdown(
        """Berikut kolom yang biasanya **paling kuat terkait churn** pada dataset Telco:

- **tenure**: pelanggan baru (tenure rendah) lebih berisiko churn.
- **Contract**: *Month-to-month* paling rawan; kontrak 1–2 tahun menekan churn.
- **PaymentMethod**: *Electronic check* sering churn lebih tinggi dibanding metode otomatis.
- **InternetService**: *Fiber optic* cenderung churn lebih tinggi dibanding DSL / No internet.
- **OnlineSecurity** & **TechSupport**: jika *No*, churn naik.
- **MonthlyCharges**: tagihan bulanan tinggi cenderung meningkatkan churn.
- **PaperlessBilling**: *Yes* sering terkait churn lebih tinggi di data ini.

Di bawah ini ada indikasi dari model (jika model adalah Logistic Regression)."""
    )
    coef_df = top_coefficients()
    if coef_df is not None:
        st.markdown("**Indikasi dari model (top koefisien):**")
        st.dataframe(coef_df, use_container_width=True)
    else:
        st.info("Model tidak menyediakan `coef_` (bukan Logistic Regression), jadi koefisien tidak ditampilkan.")

st.divider()

with st.form("single_form"):
    st.subheader("Input Customer")

    r1, r2, r3, r4 = st.columns(4)
    gender = r1.selectbox("Gender", CATEGORIES["gender"], index=0)
    senior = r2.radio("Senior Citizen", ["No", "Yes"], horizontal=True, index=0)
    partner = r3.selectbox("Partner", CATEGORIES["Partner"], index=0)
    dependents = r4.selectbox("Dependents", CATEGORIES["Dependents"], index=0)

    st.markdown("### Layanan")
    c5, c6, c7 = st.columns(3)
    phone = c5.selectbox("PhoneService", CATEGORIES["PhoneService"], index=1)
    if phone == "No":
        multiple = "No phone service"
        c6.selectbox("MultipleLines", CATEGORIES["MultipleLines"], index=2, disabled=True)
    else:
        multiple = c6.selectbox("MultipleLines", ["No", "Yes"], index=0)

    internet = c7.selectbox("InternetService", CATEGORIES["InternetService"], index=0)

    st.markdown("### Add-on Internet (opsional)")
    c8, c9, c10, c11 = st.columns(4)
    if internet == "No":
        online_security = "No internet service"
        online_backup = "No internet service"
        device_prot = "No internet service"
        tech_support = "No internet service"
        c8.selectbox("OnlineSecurity", CATEGORIES["OnlineSecurity"], index=2, disabled=True)
        c9.selectbox("OnlineBackup", CATEGORIES["OnlineBackup"], index=2, disabled=True)
        c10.selectbox("DeviceProtection", CATEGORIES["DeviceProtection"], index=2, disabled=True)
        c11.selectbox("TechSupport", CATEGORIES["TechSupport"], index=2, disabled=True)
    else:
        online_security = c8.selectbox("OnlineSecurity", ["No", "Yes"], index=0)
        online_backup = c9.selectbox("OnlineBackup", ["No", "Yes"], index=0)
        device_prot = c10.selectbox("DeviceProtection", ["No", "Yes"], index=0)
        tech_support = c11.selectbox("TechSupport", ["No", "Yes"], index=0)

    c12, c13 = st.columns(2)
    if internet == "No":
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"
        c12.selectbox("StreamingTV", CATEGORIES["StreamingTV"], index=2, disabled=True)
        c13.selectbox("StreamingMovies", CATEGORIES["StreamingMovies"], index=2, disabled=True)
    else:
        streaming_tv = c12.selectbox("StreamingTV", ["No", "Yes"], index=0)
        streaming_movies = c13.selectbox("StreamingMovies", ["No", "Yes"], index=0)

    st.markdown("### Kontrak & Pembayaran")
    c14, c15, c16 = st.columns(3)
    contract = c14.selectbox("Contract", CATEGORIES["Contract"], index=0)
    paperless = c15.selectbox("PaperlessBilling", CATEGORIES["PaperlessBilling"], index=1)
    payment = c16.selectbox("PaymentMethod", CATEGORIES["PaymentMethod"], index=0)

    st.markdown("### Charges")
    c17, c18, c19 = st.columns(3)
    tenure = c17.number_input("Tenure (bulan)", min_value=0, max_value=120, value=12, step=1)
    monthly = c18.number_input("MonthlyCharges", min_value=0.0, value=70.0, step=0.5)
    auto_total = c19.checkbox("Auto TotalCharges = tenure * MonthlyCharges", value=True)

    if auto_total:
        total = float(tenure) * float(monthly)
        st.caption(f"TotalCharges otomatis: {total:.2f}")
    else:
        total = st.number_input("TotalCharges (manual)", min_value=0.0, value=float(tenure) * float(monthly), step=1.0)

    submitted = st.form_submit_button("🔍 Prediksi")

if submitted:
    raw_input = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_prot,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    pred, proba, X_encoded = predict_one(raw_input)

    st.subheader("Hasil Prediksi")
    left, right = st.columns([1, 2])

    with left:
        if pred == 1:
            st.error("⚠️ Prediksi: **CHURN (Yes)**")
        else:
            st.success("✅ Prediksi: **NOT CHURN (No)**")
        st.metric("Probabilitas churn", f"{proba:.3f}")
        st.caption("Threshold tetap = 0.50")

    with right:
        st.markdown("**Input (raw)**")
        st.dataframe(pd.DataFrame([raw_input]), use_container_width=True)

        st.markdown("**Fitur setelah encoding (untuk model)**")
        st.dataframe(X_encoded, use_container_width=True)
