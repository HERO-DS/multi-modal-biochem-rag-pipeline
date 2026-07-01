# src/app.py
import streamlit as st
import requests

st.set_page_config(page_title="Biochemical RAG Hub", layout="wide", page_icon="🧬")

st.markdown("## ⚙️ Backend Service Mapping")
backend_url = st.text_input("FastAPI Endpoint URL", value="http://127.0.0.1:8001/api/v1/predict")

st.markdown("---")
st.markdown("# 🧬 Multi-Modal Biochemical Predictive Engine")
st.caption("Welcome to the RAG Serving Layer Dashboard. Extract deep embedding structures via ChemBERTa, evaluate blood-brain barrier (BBB) properties, and pull structural vector space analogs in real-time.")

left_column, right_column = st.columns([1, 1])

with left_column:
    st.markdown("### 🔬 Target Molecular Input")
    smiles_input = st.text_input("Enter SMILES String", value="CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    
    if st.button("Run Pipeline Inference", type="primary"):
        with st.spinner("Processing multi-modal extraction channels..."):
            try:
                payload = {"smiles": smiles_input}
                response = requests.post(backend_url, json=payload)
                
                if response.status_code == 200:
                    st.session_state["inference_data"] = response.json()
                    st.success("Analysis complete!")
                else:
                    st.error(f"Backend Server returned an error: {response.text}")
            except Exception as e:
                st.error(f"Could not connect to service endpoint: {e}")

if "inference_data" in st.session_state:
    res = st.session_state["inference_data"]
    
    with left_column:
        st.markdown("### 📊 Diagnostic Output Analysis")
        m1, m2 = st.columns(2)
        m1.metric("Estimated Partition Coefficient (logBB)", f"{res['logBB_prediction']:.4f}")
        m2.metric("Permeability Confidence", f"{res['permeability_probability']*100:.2f}%")
        
        if res['is_permeable']:
            st.success("✅ Model Prediction: This molecule is classified as BBB+ (Permeable).")
        else:
            st.error("🚫 Model Prediction: This molecule is classified as BBB- (Non-permeable).")
            
        st.markdown("#### 🧬 ChemBERTa Vector Profile Extract")
        st.json(res['embedding_sample'])

    with right_column:
        st.markdown("### 📚 Retrieved Vector Space Analogs (RAG Context)")
        st.info("Top structural analogs found via ChromaDB Vector Space Match:")
        
        # 1. Render the vector database matches cleanly
        st.text_area(
            label="ChromaDB Knowledge Retrieval Data Payload", 
            value=res.get('rag_historical_context', 'No context returned.'), 
            height=200
        )
        
        st.markdown("---")
        
        # 2. Render the actual LLM Synthesis Report markdown
        st.markdown("### 📝 Automated Biochemical Evaluation Report")
        st.markdown(res.get('llm_synthesis_report', '⚠️ No LLM synthesis report found in response.'))