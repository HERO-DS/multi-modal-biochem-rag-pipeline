# app.py
import streamlit as st
import requests

# 1. Page Configurations
st.set_page_config(
    page_title="Biochem Multi-Modal RAG Platform",
    page_icon="🧬",
    layout="wide"
)

# 2. Main Header App Styling
st.title("🧬 Multi-Modal Biochemical Predictive Engine")
st.markdown("""
Welcome to the RAG Serving Layer Dashboard. Enter a raw **SMILES** string sequence below 
to extract deep embedding structures via **ChemBERTa** and evaluate blood-brain barrier (BBB) properties in real-time.
""")
st.write("---")

# 3. Create Sidebar for System Status Information
st.sidebar.header("⚙️ Backend Service Mapping")
backend_url = st.sidebar.text_input("FastAPI Endpoint URL", value="http://127.0.0.1:8000/api/v1/predict")

st.sidebar.markdown("---")
st.sidebar.info("""
**Pipeline Metrics Status:**
* Embeddings Layer: Active (768-D)
* Regression Head: Random Forest
* Classification Head: Logistic Regression
""")

# 4. Main UI Input Layout Split
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🔬 Target Molecular Input")
    # Sample SMILES container presets
    smiles_input = st.text_input("Enter SMILES String", value="CCO", help="e.g., CCO for Ethanol, CN1C=NC2=C1C(=O)N(C(=O)N2C)C for Caffeine")
    
    submit_btn = st.button("Run Pipeline Inference", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Diagnostic Output Analysis")
    
    if submit_btn:
        with st.spinner("Streaming data through multi-modal model heads..."):
            try:
                # Pack the payload exactly matching the FastAPI SmilesPayload Schema
                payload = {"smiles": smiles_input}
                response = requests.post(backend_url, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Distribute numerical predictions into clean UI metric blocks
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric(
                            label="Estimated Partition Coefficient (logBB)", 
                            value=f"{data['logBB_prediction']:.4f}"
                        )
                    with m2:
                        prob_percentage = data['permeability_probability'] * 100
                        st.metric(
                            label="Permeability Confidence", 
                            value=f"{prob_percentage:.2f}%"
                        )
                    
                    st.write("---")
                    
                    # Highlight Classification State using native Alert Layouts
                    if data['is_permeable']:
                        st.success("🧠 **Model Prediction:** This molecule is classified as **BBB+** (Permeable to the Blood-Brain Barrier).")
                    else:
                        st.error("🚫 **Model Prediction:** This molecule is classified as **BBB-** (Non-permeable / Blocked by the Barrier).")
                        
                    # 5. Diagnostic Vector Visualization
                    st.markdown("### 🧬 ChemBERTa Vector Profile Extract")
                    st.markdown("Below are the first 5 dimensions extracted from the frozen 768-D contextual attention layer:")
                    st.json(data['embedding_sample'])
                    
                else:
                    st.error(f"Backend Server Error (Code {response.status_code}): {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Connection Refused! Make sure your FastAPI backend server is actively running (`uvicorn src.serving_app:app --reload`) on port 8000.")
    else:
        st.info("💡 Standby. Enter a SMILES code configuration and press 'Run Pipeline Inference' to start processing.")