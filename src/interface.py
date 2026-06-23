import os
import streamlit as st
import requests
from rdkit import Chem
from rdkit.Chem import Draw
import base64
import pandas as pd

st.set_page_config(
    page_title="Neuro-Biochem RAG Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.markdown(
    """
    <style>
    .reportview-container { background-color: #fcfcfc; }
    .stButton>button {
        width: 100%;
        background-color: #4A90E2;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        border: none;
        padding: 10px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #357ABD;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Neuro-Chemical Blood-Brain Barrier (BBB) Penetration Dashboard")
st.markdown("---")

sidebar_smiles = st.sidebar.text_input("🔬 Enter Target Compound SMILES:", "CCN(CC)CC")
api_endpoint = "http://127.0.0.1:8000/predict"

tabs = st.tabs(["🔮 Real-Time Inference", "📉 Chemical Embedding Manifolds", "📈 Model Diagnostics"])

def render_svg_molecule(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        drawer = Draw.MolDraw2DSVG(400, 250)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        b64 = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
        return f'<img src="data:image/svg+xml;base64,{b64}"/>'
    except Exception:
        return None

# =====================================================================
# TAB 1: REAL-TIME INFERENCE GATEWAY
# =====================================================================
with tabs[0]:
    st.header("Automated Compound Vector Classification")
    if st.sidebar.button("Run Prediction Sequence", key="predict_btn"):
        with st.spinner("Processing molecular features and running model inference..."):
            try:
                response = requests.post(api_endpoint, json={"canonical_smiles": sidebar_smiles})
                if response.status_code == 200:
                    data = response.json()
                    
                    # 1. Molecular Structure Layout Row
                    st.subheader("Molecular Structure Layout")
                    mol_svg = render_svg_molecule(sidebar_smiles)
                    if mol_svg:
                        st.markdown(
                            f'<div style="background: white; padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0; display: inline-block; text-align: center; margin-bottom: 20px;">{mol_svg}</div>', 
                            unsafe_allow_html=True
                        )
                    else:
                        st.warning("⚠️ Could not render 2D molecular layout.")
                    
                    # 2. Pipeline Model Inference metrics
                    st.subheader("Classification & Partition Metrics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        status = "✅ PERMEABLE (BBB+)" if data["is_bbb_permeable"] == 1 else "❌ BLOCKED (BBB-)"
                        color = "#2ecc71" if data["is_bbb_permeable"] == 1 else "#e74c3c"
                        st.markdown(
                            f"""
                            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid {color}; min-height: 95px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                                <p style="margin: 0; font-size: 11px; color: #7f8c8d; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">Blood-Brain Barrier Status</p>
                                <h4 style="margin: 8px 0 0 0; color: {color}; font-size: 16px; font-weight: bold; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;" title="{status}">{status}</h4>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    with col2:
                        st.markdown(
                            f"""
                            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #9b59b6; min-height: 95px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                                <p style="margin: 0; font-size: 11px; color: #7f8c8d; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">Inference Confidence</p>
                                <h4 style="margin: 8px 0 0 0; color: #2c3e50; font-size: 18px; font-weight: bold;">{data['permeability_confidence'] * 100:.2f}%</h4>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    with col3:
                        st.markdown(
                            f"""
                            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #3498db; min-height: 95px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                                <p style="margin: 0; font-size: 11px; color: #7f8c8d; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">Predicted Partition (logBB)</p>
                                <h4 style="margin: 8px 0 0 0; color: #2c3e50; font-size: 18px; font-weight: bold;">{data['predicted_logbb']:.4f}</h4>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )

                    st.markdown("<br>", unsafe_allow_html=True)
                        
                    # 3. Classical biochemical indicators
                    st.subheader("Computed Biochemical Structural Metrics")
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.markdown(
                            f"""
                            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #e67e22; min-height: 95px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                                <p style="margin: 0; font-size: 11px; color: #7f8c8d; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">Molecular Weight</p>
                                <h4 style="margin: 8px 0 0 0; color: #2c3e50; font-size: 18px; font-weight: bold;">{data['molecular_weight']:.2f} <span style="font-size: 12px; font-weight: normal; color: #7f8c8d;">g/mol</span></h4>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    with col_m2:
                        st.markdown(
                            f"""
                            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #1abc9c; min-height: 95px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                                <p style="margin: 0; font-size: 11px; color: #7f8c8d; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">Calculated Partition Coefficient (LogP)</p>
                                <h4 style="margin: 8px 0 0 0; color: #2c3e50; font-size: 18px; font-weight: bold;">{data['log_p']:.4f}</h4>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # 4. RAG CONTEXTUAL REPORT VIEW
                    st.subheader("🧠 Knowledge Base RAG Analysis Report")
                    st.info(data["rag_summary"])
                    
                    st.markdown("##### Structurally Similar Reference Compounds (Training Store Matches)")
                    neighbors_df = pd.DataFrame(data["nearest_neighbors"])
                    if not neighbors_df.empty:
                        neighbors_df.columns = ["Molecule ID", "SMILES Sequence", "Vector Distance (L2)"]
                        st.dataframe(neighbors_df, use_container_width=True)

                else:
                    st.error(f"Backend API failure. Status code: {response.status_code}")
            except Exception as e:
                st.error(f"Could not reach FastAPI engine server: {str(e)}")
    else:
        st.info("Provide a valid molecular SMILES sequence in the sidebar and click 'Run Prediction Sequence'.")

# =====================================================================
# TAB 2: MANIFOLD EMBEDDING MAPS
# =====================================================================
with tabs[1]:
    st.header("High-Dimensional Chemical Vector Spaces")
    st.markdown("Visualizing how the transformer embeddings map similarities in global and local manifolds.")
    
    col_pca, col_tsne = st.columns(2)
    with col_pca:
        st.subheader("Global Variational Linear Space (PCA)")
        pca_img = "reports/chemical_space_clustering_pca.png"
        if os.path.exists(pca_img):
            st.image(pca_img, use_column_width=True)
        else:
            st.caption("PCA visualization plot missing. Run src/visualizer.py to generate.")

    with col_tsne:
        st.subheader("Local Neighborhood Manifolds (t-SNE)")
        tsne_img = "reports/chemical_space_clustering_tsne.png"
        if os.path.exists(tsne_img):
            st.image(tsne_img, use_column_width=True)
        else:
            st.caption("t-SNE visualization plot missing. Run src/visualizer.py to generate.")

# =====================================================================
# TAB 3: PERFORMANCE DIAGNOSTICS REPORT
# =====================================================================
with tabs[2]:
    st.header("Dual-Head Structural Performance Logs")
    perf_img = "reports/model_performance_diagnostics.png"
    if os.path.exists(perf_img):
        st.image(perf_img, use_column_width=True)
    else:
        st.caption("Performance diagnostics curves missing. Run src/visualizer.py to generate.")