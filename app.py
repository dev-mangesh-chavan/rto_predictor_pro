import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import io

# ==========================================
# 1. PAGE CONFIGURATION & DARK THEME
# ==========================================
st.set_page_config(
    page_title="RTO Predictor Pro | Enterprise ML",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Minimal, Safe CSS for Dark Theme & Clean UI
st.markdown("""
    <style>
    /* Pull the top container up so title is higher */
    div.block-container {
        padding-top: 1.5rem !important;
    }
    
    /* Dark Theme Background */
    .stApp { background-color: #0E1117; }
    
    /* Make standard text slightly larger for video readability */
    p, li, .stMarkdown {
        font-size: 1.15rem !important;
    }
    
    /* Gradient Text for Main Title */
    .gradient-text {
        background: -webkit-linear-gradient(45deg, #FF3F6C, #FF7E5F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4.2rem;
        font-weight: 900;
        text-align: center;
        padding-top: 0rem;
        margin-top: -1.5rem;
        margin-bottom: 0;
    }
    
    .sub-text {
        text-align: center;
        color: #A0AEC0;
        font-size: 1.5rem;
        letter-spacing: 2px;
        margin-bottom: 2rem;
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HEADER
# ==========================================
st.markdown("<div class='gradient-text'>RTO PREDICTOR PRO</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>ENTERPRISE ML PIPELINE FOR E-COMMERCE</div>", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000"

# ==========================================
# 3. TABS
# ==========================================
tab_arch, tab_exec = st.tabs(["🏗️ ARCHITECTURE & PERFORMANCE", "🚀 LIVE AI EXECUTION"])

# ------------------------------------------
# TAB 1: ARCHITECTURE & REAL METRICS
# ------------------------------------------
with tab_arch:
    st.markdown("### 🚨 The Business Problem")
    st.info("In the E-commerce sector, **30-40% of Cash-on-Delivery (COD) orders** result in Return to Origin (RTO). This drains millions in reverse logistics, wasted packaging, and blocked inventory. Manual verification at scale is impossible.")
    
    st.markdown("---")
    st.markdown("### 🧠 The AI Solution & Real Performance")
    st.write("We built and trained a robust **Random Forest Classifier** to predict RTO risk in milliseconds before the package is shipped.")
    
    # PROUDLY DISPLAYING YOUR ACTUAL MODEL METRICS (Updated for 10k data / 2k test)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(label="Model Accuracy", value="82.30%", delta="Highly Accurate")
    c2.metric(label="ROC-AUC Score", value="81.63%", delta="Excellent Pattern Detection")
    c3.metric(label="Precision (RTO)", value="89.00%", delta="Very Low False Positives")
    c4.metric(label="Test Dataset", value="2,000 Orders", delta="Unseen Data") 
    
    st.success("**Business Impact:** With 89% precision, when our AI flags an order as 'High Risk', it is correct 89% of the time. This allows the business to confidently ask for advance payments without annoying genuine customers.")
    
    st.markdown("---")
    st.markdown("### 🔄 System Workflow (Millisecond Execution)")
    
    # Clean, Native Streamlit workflow
    w1, w2, w3, w4 = st.columns(4)
    with w1:
        st.warning("**1. API Intercept**\n\nFastAPI catches live checkout data.")
    with w2:
        st.info("**2. Processing**\n\nDataProcessor scales & cleans instantly.")
    with w3:
        st.success("**3. ML Engine**\n\nRandom Forest calculates Risk %.")
    with w4:
        st.error("**4. Decision**\n\nOrder is Auto-Approved or Blocked.")

# ------------------------------------------
# TAB 2: LIVE EXECUTION
# ------------------------------------------
with tab_exec:
    st.write("Test the end-to-end pipeline. The UI talks directly to the FastAPI backend.")
    
    sub_single, sub_batch = st.tabs(["👤 SINGLE ORDER INFERENCE", "📦 WAREHOUSE BATCH SCAN"])
    
    # --- SINGLE ORDER ---
    with sub_single:
        st.markdown("#### Simulate Live Checkout")
        
        profiles = {
            "Select a test case...": None,
            "🔴 TestCase A: High Risk (Serial Returner, COD)": {"Cart_Value": 12500, "Return_Rate": 0.85, "Pincode_Tier": 3, "Address_Quality": 2, "Is_COD": "Yes"},
            "🟢 TestCase B: Safe (Loyal User, Prepaid)": {"Cart_Value": 2500, "Return_Rate": 0.05, "Pincode_Tier": 1, "Address_Quality": 9, "Is_COD": "No"},
            "🟡 TestCase C: Borderline (Impulse Buyer)": {"Cart_Value": 6000, "Return_Rate": 0.40, "Pincode_Tier": 2, "Address_Quality": 6, "Is_COD": "Yes"},
            "⚙️ Manual Entry (Type yourself)": {"Cart_Value": 1500, "Return_Rate": 0.10, "Pincode_Tier": 2, "Address_Quality": 7, "Is_COD": "Yes"}
        }
        
        selected_case = st.selectbox("Load Data Profile:", list(profiles.keys()))
        data = profiles[selected_case] if profiles[selected_case] else profiles["⚙️ Manual Entry (Type yourself)"]
        
        with st.form("single_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                cart_val = st.number_input("Cart Value (₹)", value=int(data["Cart_Value"]), step=500)
                is_cod_input = st.selectbox("Is COD?", ["Yes", "No"], index=0 if data["Is_COD"] == "Yes" else 1)
            with c2:
                ret_rate = st.slider("Past Return Rate (%)", 0, 100, int(data["Return_Rate"]*100)) / 100
                pin_tier = st.selectbox("Pincode Tier (1=Metro, 3=Rural)", [1, 2, 3], index=int(data["Pincode_Tier"])-1)
            with c3:
                addr_qual = st.slider("Address Quality (1-10)", 1, 10, int(data["Address_Quality"]))
                st.write("")
                st.write("")
                submit_btn = st.form_submit_button("🔍 PREDICT RTO RISK", use_container_width=True)

        if submit_btn:
            payload = {
                "Cart_Value": cart_val,
                "Return_Rate": ret_rate,
                "Pincode_Tier": pin_tier,
                "Address_Quality": addr_qual,
                "Is_COD": 1 if is_cod_input == "Yes" else 0
            }
            
            with st.spinner("Connecting to FastAPI..."):
                time.sleep(0.8) 
                try:
                    response = requests.post(f"{API_URL}/predict/single", json=payload)
                    if response.status_code == 200:
                        res = response.json()
                        st.markdown("### 🧠 AI Decision")
                        
                        reason = res.get("ai_reason", "AI Processed.")
                        
                        if res["status"] == "Safe":
                            st.success(f"**✅ SAFE TO SHIP** | Risk Probability: **{res['risk_score']}%**")
                            st.info(f"**Action:** {res['recommended_action']}\n\n**💡 AI Reason:** {reason}")
                        elif res["status"] == "Review":
                            st.warning(f"**⚠️ REVIEW NEEDED** | Risk Probability: **{res['risk_score']}%**")
                            st.info(f"**Action:** {res['recommended_action']}\n\n**💡 AI Reason:** {reason}")
                        else:
                            st.error(f"**🚫 BLOCKED** | Risk Probability: **{res['risk_score']}%**")
                            st.info(f"**Action:** {res['recommended_action']}\n\n**💡 AI Reason:** {reason}")
                    else:
                        st.error(f"API Error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("🚨 API Connection Failed. Run `python main.py` in your terminal.")

    # --- BATCH PROCESSING ---
    with sub_batch:
        st.markdown("#### Automated Bulk Scan")
        st.write("Upload a batch of orders to categorize them instantly via the AI Engine.")
        
        # Generating exactly 100 rows of dummy data for a realistic batch test
        np.random.seed(42)
        dummy_data = pd.DataFrame({
            "Order_ID": [f"ORD_BATCH_{str(i).zfill(3)}" for i in range(1, 101)],
            "Cart_Value": np.random.randint(500, 18000, 100),
            "Return_Rate": np.round(np.random.beta(1, 5, 100), 2),
            "Pincode_Tier": np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2], size=100),
            "Address_Quality": np.random.randint(2, 11, 100),
            "Is_COD": np.random.choice([0, 1], p=[0.4, 0.6], size=100)
        })
        
        csv_buffer = io.BytesIO()
        dummy_data.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="📥 Download Sample CSV (100 Orders)",
            data=csv_buffer.getvalue(),
            file_name="test_batch_100_orders.csv",
            mime="text/csv"
        )
        
        uploaded_file = st.file_uploader("Upload your CSV File", type=["csv"])
        
        if uploaded_file is not None:
            if st.button("🚀 PROCESS BATCH", type="primary"):
                with st.spinner("Processing massive batch via FastAPI..."):
                    time.sleep(1.5)
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                        response = requests.post(f"{API_URL}/predict/batch", files=files)
                        
                        if response.status_code == 200:
                            df_results = pd.DataFrame(response.json()["data"])
                            
                            s_count = len(df_results[df_results["Status"] == "Safe"])
                            r_count = len(df_results[df_results["Status"] == "Review"])
                            b_count = len(df_results[df_results["Status"] == "Blocked"])
                            
                            st.markdown("### 📊 Batch Sorting Summary")
                            m1, m2, m3 = st.columns(3)
                            m1.metric("🟢 Safe", f"{s_count}")
                            m2.metric("🟡 Review", f"{r_count}")
                            m3.metric("🔴 Blocked", f"{b_count}")
                            
                            st.markdown("### 📋 Detailed Action Report")
                            
                            def color_status(val):
                                if val == 'Safe': return 'color: #4ADE80; font-weight: bold;'
                                elif val == 'Review': return 'color: #FACC15; font-weight: bold;'
                                else: return 'color: #F87171; font-weight: bold;'
                            
                            display_cols = ['Order_ID', 'Cart_Value', 'Return_Rate', 'Is_COD', 'Risk_Score_%', 'Status', 'AI_Reason']
                            st.dataframe(df_results[display_cols].style.map(color_status, subset=['Status']), use_container_width=True, height=400)
                            
                        else:
                            st.error(f"Backend API Error: {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("🚨 API Connection Failed. Start `main.py` first.")