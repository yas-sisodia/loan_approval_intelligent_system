import os
import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page Configuration & Zero-Scroll Layout Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Loan Underwriter",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
        /* Lock body scroll to fit on one screen */
        html, body, [data-testid="stAppViewContainer"] {
            overflow: hidden !important;
            height: 100vh;
        }
        /* Reduce block padding */
        .block-container {
            padding-top: 1.2rem !important;
            padding-bottom: 0rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 100% !important;
        }
        /* Tighten vertical spacing */
        div[data-testid="stVerticalBlock"] > div {
            gap: 0.4rem !important;
        }
        .stSelectbox label, .stNumberInput label, .stSlider label {
            font-size: 0.80rem !important;
            font-weight: 600 !important;
            margin-bottom: -4px !important;
        }
        .stSelectbox div[data-baseweb="select"] {
            min-height: 32px !important;
            height: 32px !important;
        }
        .stNumberInput input {
            height: 32px !important;
            padding: 4px 8px !important;
        }
        /* Compact cards for financial metrics */
        .metric-box {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 8px 10px;
            text-align: center;
        }
        .metric-val {
            font-size: 1.1rem;
            font-weight: 700;
            color: #1e293b;
        }
        .metric-lbl {
            font-size: 0.70rem;
            color: #64748b;
            text-transform: uppercase;
        }
        /* Button styling */
        div.stButton > button {
            width: 100%;
            height: 38px;
            font-weight: 600;
            margin-top: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Pipeline Ingestion
# ---------------------------------------------------------
# MODEL_FILE = "best_loan_model_pipeline.joblib"

# @st.cache_resource
# def load_pipeline(path: str):
#     if not os.path.exists(path):
#         return None
#     return joblib.load(path)

# pipeline = load_pipeline(MODEL_FILE)

# if pipeline is None:
#     st.error(f"Missing '{MODEL_FILE}'. Please verify the serialized artifact is in the working directory.")
#     st.stop()
MODEL_FILE = "best_loan_model_pipeline.joblib"

@st.cache_resource
def load_pipeline(path: str):
    if not os.path.exists(path):
        import train
        train.train_and_export()
    
    try:
        return joblib.load(path)
    except Exception:
        # If unpickling fails due to version mismatch, retrain fresh
        import train
        train.train_and_export()
        return joblib.load(path)

pipeline = load_pipeline(MODEL_FILE)

# ---------------------------------------------------------
# Main Single-Screen Split Interface
# ---------------------------------------------------------
left_col, right_col = st.columns([1.1, 0.9], gap="medium")

# --- LEFT COLUMN: Input Matrix ---
with left_col:
    st.markdown("### 📝 Application Profile")
    
    # Row 1: Financials
    r1_1, r1_2, r1_3 = st.columns(3)
    with r1_1:
        app_income = st.number_input("Applicant Income ($/mo)", 0, 150000, 5000, 250)
    with r1_2:
        coapp_income = st.number_input("Co-app Income ($/mo)", 0.0, 100000.0, 1500.0, 250.0)
    with r1_3:
        loan_k = st.number_input("Loan Amount ($k)", 5.0, 2000.0, 150.0, 5.0)

    # Row 2: Facility & History
    r2_1, r2_2, r2_3 = st.columns(3)
    with r2_1:
        term_map = {"30 Yrs (360m)": 360.0, "15 Yrs (180m)": 180.0, "20 Yrs (240m)": 240.0, "5 Yrs (60m)": 60.0}
        term_label = st.selectbox("Facility Term", list(term_map.keys()), index=0)
        loan_term = term_map[term_label]
    with r2_2:
        credit_choice = st.selectbox("Credit History", ["Clean Record (1.0)", "Default / Derogatory (0.0)"])
        credit_history = 1.0 if "Clean" in credit_choice else 0.0
    with r2_3:
        prop_area = st.selectbox("Property Area", ["Semiurban", "Urban", "Rural"])

    # Row 3: Demographics
    r3_1, r3_2, r3_3, r3_4 = st.columns(4)
    with r3_1:
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    with r3_2:
        married = st.selectbox("Married", ["Yes", "No"])
    with r3_3:
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    with r3_4:
        self_emp = st.selectbox("Self Employed", ["No", "Yes"])

    # Row 4: Gender
    gender = st.selectbox("Applicant Gender", ["Male", "Female"])

    # Action Button
    assess_button = st.button("Evaluate Application", type="primary")

# --- RIGHT COLUMN: Policy & Evaluation Output ---
with right_col:
    st.markdown("### 🎯 Decision & Risk Assessment")
    
    threshold = st.slider("Underwriting Threshold Cutoff", 0.30, 0.80, 0.55, 0.05)

    if assess_button:
        # Build payload
        payload = pd.DataFrame([{
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_emp,
            "ApplicantIncome": app_income,
            "CoapplicantIncome": coapp_income,
            "LoanAmount": loan_k,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": prop_area
        }])

        # # Inference
        # proba = pipeline.predict_proba(payload)[0, 1]
        # approved = proba >= threshold

        # # Financial Calculations
        # total_income = app_income + coapp_income
        # nominal_loan = loan_k * 1000
        # monthly_emi = (nominal_loan / loan_term) if loan_term > 0 else 0
        # dti = (monthly_emi / total_income * 100) if total_income > 0 else 999.0

        # # Decision Output
        # if approved:
        #     st.success(f"**DECISION: APPROVED** — P(Approval): **{proba*100:.1f}%** (≥ {threshold:.2f} threshold)")
        # else:
        #     st.error(f"**DECISION: REJECTED** — P(Approval): **{proba*100:.1f}%** (< {threshold:.2f} threshold)")

        # 1. Pipeline ML Inference
        proba = pipeline.predict_proba(payload)[0, 1]
        approved = proba >= threshold

        # 2. Financial Sanity Calculations
        total_income = app_income + coapp_income
        nominal_loan = loan_k * 1000
        monthly_emi = (nominal_loan / loan_term) if loan_term > 0 else 0
        dti = (monthly_emi / total_income * 100) if total_income > 0 else 999.0

        # 3. Hard Underwriting Policy Override (Guardrail)
        override_reason = None
        if dti > 45.0:
            approved = False
            override_reason = f"DTI of {dti:.1f}% exceeds institutional ceiling (45.0%)."
        elif total_income < 1000:
            approved = False
            override_reason = f"Household income (${total_income:,.0f}) is below minimum service threshold ($1,000)."

        # 4. Final Decision Display
        if approved:
            st.success(f"**DECISION: APPROVED** — P(Approval): **{proba*100:.1f}%** (≥ {threshold:.2f} threshold)")
        else:
            if override_reason:
                st.error(f"**DECISION: REJECTED (POLICY OVERRIDE)** — {override_reason}")
            else:
                st.error(f"**DECISION: REJECTED** — P(Approval): **{proba*100:.1f}%** (< {threshold:.2f} threshold)")

        st.progress(float(proba))

        # Financial Indicator Cards
        st.markdown(
            f"""
            <div style="display: flex; gap: 8px; margin-top: 4px; margin-bottom: 6px;">
                <div class="metric-box" style="flex: 1;">
                    <div class="metric-val">${total_income:,.0f}</div>
                    <div class="metric-lbl">Household Income</div>
                </div>
                <div class="metric-box" style="flex: 1;">
                    <div class="metric-val">${monthly_emi:,.0f}/mo</div>
                    <div class="metric-lbl">Estimated EMI</div>
                </div>
                <div class="metric-box" style="flex: 1;">
                    <div class="metric-val">{dti:.1f}%</div>
                    <div class="metric-lbl">DTI Ratio</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Underwriting Diagnostic
        if credit_history == 0.0:
            st.warning("⚠️ **High Risk Flag:** Recorded history of default/delinquency.")
        elif dti > 45.0:
            st.warning(f"⚠️ **High DTI Alert:** EMI consumes {dti:.1f}% of income (> 45% standard limit).")
        else:
            st.info("✅ **Clean Underwriting Check:** Credit rating and DTI within standard risk parameters.")

    else:
        # Default placeholder before button click
        st.info("👈 Enter profile parameters on the left and click **Evaluate Application** to run inference.")