# Telco Customer Churn: Analysis, Prediction & Early Warning System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nusaneuron-telco-customer-churn-prediction.7jwomn.easypanel.host/)
[![Tableau Dashboard](https://img.shields.io/badge/Tableau-Public-blue)](https://public.tableau.com/views/TelcoCustomerChurnDashboard_17701711006980/Dashboard1)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

**Final Project – Data Science & Machine Learning Program**  
**Institution:** Purwadhika Digital School  
**Group Beta:**
- Asma Nur Ummah  
- Aldythya Nugraha  
- Indah Rachmadanti  

Repository: https://github.com/appaldyt/Telco-Customer-Churn-Prediction

---

## Project Overview

Industri telekomunikasi memiliki persaingan yang ketat. Ketika pelanggan berhenti berlangganan (*churn*), perusahaan berpotensi kehilangan pendapatan berulang dan perlu mengeluarkan biaya lebih besar untuk akuisisi pelanggan baru.  
Project ini membangun **model prediksi churn** (Yes/No) sekaligus **early warning system** yang membantu tim bisnis melakukan intervensi retensi lebih cepat dan tepat sasaran.

---

## Dataset

Dataset yang digunakan adalah *Telco Customer Churn* dengan:
- **7,043** pelanggan
- **21** kolom (20 fitur + target `Churn`)
- **Churn rate:** ~**26.54%** (Yes)

**Fitur utama**:
- Demografi: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- Layanan: `InternetService`, `OnlineSecurity`, `TechSupport`, `StreamingTV`, dll.
- Kontrak & billing: `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`
- Biaya: `MonthlyCharges`, `TotalCharges` 

---

## Business Objective & Metric

### Objective
1. Memprediksi pelanggan berisiko churn (klasifikasi biner).
2. Memberikan dasar segmentasi risiko untuk strategi retensi.

### Why F2-Score?
Dalam konteks churn, **False Negative (FN)** (pelanggan churn tapi diprediksi tidak churn) biasanya lebih “mahal” karena bisnis kehilangan kesempatan retensi.  
Karena itu, evaluasi difokuskan ke **F2-score (β=2)** yang memberi bobot lebih besar pada **Recall** dibanding Precision.

---

## Methodology

### 1) Data Loading & Cleaning
- Konversi tipe data (mis. `TotalCharges` → numerik).
- Menangani nilai kosong/invalid (umumnya muncul saat `tenure = 0`).

### 2) Exploratory Data Analysis (EDA)
- Distribusi target `Churn` (Yes/No).
- Boxplot `tenure`, `MonthlyCharges`, `TotalCharges` terhadap `Churn` (menampilkan outlier).
- Visualisasi churn rate pada seluruh kolom kategorikal dengan stacked bar (proporsi churn per kategori).

### 3) Preprocessing
- One-hot encoding menggunakan `pd.get_dummies(..., drop_first=True)`.
- Split data train/test dengan stratified sampling.
- Scaling fitur numerik menggunakan `StandardScaler`.

### 4) Modeling (Benchmark)
Model yang diuji:
- Logistic Regression (class_weight='balanced')
- Random Forest (balanced)
- KNN
- Decision Tree (balanced)
- Gradient Boosting

### 5) Cross Validation
- 5-Fold Cross Validation untuk membandingkan stabilitas model berdasarkan F2-score.

### 6) Hyperparameter Tuning
- `GridSearchCV` pada model terbaik dari CV (Logistic Regression), mencari nilai `C` terbaik.

### 7) Model Interpretability (SHAP)
- Menggunakan **SHAP** untuk memahami fitur yang mendorong churn naik/turun (summary plot kiri-kanan).

---

## Key Results

### Cross Validation (F2 mean)
- **Logistic Regression (Balanced): ~0.725**
- Gradient Boosting: ~0.539
- KNN: ~0.531
- Random Forest (Balanced): ~0.503
- Decision Tree (Balanced): ~0.489

### Final Model
- **Model:** Logistic Regression (Balanced)
- **Best Params:** `C=1`, `penalty='l2'`
- **F2-score (Train): ~0.731**
- **F2-score (Test): ~0.703**

### Actionable Insights
- **Contract Migration** : Insentif bagi pelanggan month-to-month untuk beralih ke kontrak jangka panjang
- **Early Tenure Loyalty** : Program loyalitas progresif (reward bulan ke-3, 6, 12) dan proactive support untuk menjaga retensi pelanggan baru.
- **Service Quality** : Evaluasi teknis dan penyesuaian harga layanan Fiber Optic, serta penambahan bundling konten digital (Streaming).
- **Cost Efficiency** : Alokasi anggaran promosi yang terfokus hanya pada pelanggan kategori High-Risk dan High-Value

---

## Deployment

### Streamlit App (Prediction)
Aplikasi web untuk prediksi churn per-customer:
- Input data pelanggan via form (sidebar)
- Output prediksi (Churn / Not Churn) + probabilitas
- Menampilkan fitur setelah encoding (untuk transparansi)

**Live App:** https://nusaneuron-telco-customer-churn-prediction.7jwomn.easypanel.host/

### Tableau Dashboard (Monitoring)
Dashboard interaktif untuk monitoring churn:
- KPI churn rate dan total customer
- Churn rate berdasarkan `Contract`, `InternetService`, `PaymentMethod`, dll.
- Eksplorasi segmentasi pelanggan untuk keputusan retensi

**Tableau Public:** https://public.tableau.com/views/TelcoCustomerChurnDashboard_17701711006980/Dashboard1

---

## Repository Structure

```text
.
├── .streamlit/
├── artifacts/
│   ├── best_model.joblib
│   ├── scaler.joblib
│   └── feature_columns.joblib
├── dataset/
│   └── Dataset-Telco-Customer-Churn.csv
├── Beta_Analytics_Telco_Customer_Churn.ipynb
├── app.py
├── requirements.txt
├── Dockerfile
├── start.sh
├── telco_customer_churn_dashboard.twbx
└── telco_churn_artifacts.zip
```

---

## Installation & Local Run

### 1) Clone repository
```bash
git clone https://github.com/appaldyt/Telco-Customer-Churn-Prediction.git
cd Telco-Customer-Churn-Prediction
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run Streamlit
```bash
streamlit run app.py
```

---

## Docker (Optional)

```bash
docker build -t telco-churn .
docker run -p 8501:8501 telco-churn
```

---

## Notes
- File model (`artifacts/`) disediakan siap pakai. Jika file artifact tidak ditemukan, aplikasi akan mencoba mengekstrak dari `telco_churn_artifacts.zip`.

---

## Acknowledgements
- Purwadhika Digital School – Data Science & Machine Learning Program
- IBM Telco Customer Churn (public dataset)

