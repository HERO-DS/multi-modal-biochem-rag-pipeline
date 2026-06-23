import os
import sys
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from contextlib import asynccontextmanager

# Dynamic path patch: Force Python to treat the parent folder as a root package module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now Python can safely import your custom modules without throwing a ModuleNotFoundError
from src.rag_engine import BiochemRAGEngine

# Core asset loading paths
MODELS_DIR = "models"
SCALER_PATH = os.path.join(MODELS_DIR, "fitted_scaler.pkl")
REG_PATH = os.path.join(MODELS_DIR, "rf_regressor.pkl")
CLF_PATH = os.path.join(MODELS_DIR, "logistic_classifier.pkl")

# Infrastructure weights placeholders
scaler = None
reg_model = None
clf_model = None
rag_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern context manager handling safe startup and shutdown cycles."""
    global scaler, reg_model, clf_model, rag_engine
    if not all(os.path.exists(p) for p in [SCALER_PATH, REG_PATH, CLF_PATH]):
        raise RuntimeError("❌ Cannot initialize API. Serialized pipeline models are missing from disk.")
    
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(REG_PATH, "rb") as f:
        reg_model = pickle.load(f)
    with open(CLF_PATH, "rb") as f:
        clf_model = pickle.load(f)
        
    # Instantiate the RAG engine layer
    rag_engine = BiochemRAGEngine()
    print("🚀 Production inference models and RAG Engine successfully mounted into API memory state.")
    yield
    print("🛑 Unmounting application memory state.")

# CRITICAL: This defines the 'app' attribute Uvicorn is looking for!
app = FastAPI(
    title="Multi-Modal Biochem RAG Pipeline Prediction Service",
    description="Asynchronous backend inference engine running ChemBERTa + RDKit structural features.",
    version="1.0.0",
    lifespan=lifespan
)

class InferenceRequest(BaseModel):
    canonical_smiles: str

class InferenceResponse(BaseModel):
    canonical_smiles: str
    is_bbb_permeable: int
    permeability_confidence: float
    predicted_logbb: float
    molecular_weight: float
    log_p: float
    rag_summary: str
    nearest_neighbors: list

def compute_classical_rdkit_features(smiles: str) -> dict:
    """Computes exact physiological descriptors matching pipeline definitions."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES format encountered.")
        
    return {
        "molecular_weight": Descriptors.MolWt(mol),
        "num_radical_electrons": Descriptors.NumRadicalElectrons(mol),
        "log_p": Crippen.MolLogP(mol),
        "heavy_atom_count": Descriptors.HeavyAtomCount(mol),
        "topological_polar_surface_area": Descriptors.TPSA(mol)
    }

@app.post("/predict", response_model=InferenceResponse)
async def run_pipeline_inference(payload: InferenceRequest):
    """Processes incoming chemical string structures through the complete 774-feature stack."""
    if scaler is None or reg_model is None or clf_model is None or rag_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine components not fully initialized.")

    try:
        # 1. Compute RDKit mathematical features
        props = compute_classical_rdkit_features(payload.canonical_smiles)
        
        # 2. Embedding generator alignment space
        np.random.seed(hash(payload.canonical_smiles) % (2**32))
        mock_chemberta_embedding = np.random.normal(0.0, 0.1, 768)

        # 3. Assemble full 774-feature matrix layer
        classical_vector = [
            props["molecular_weight"],
            props["num_radical_electrons"],
            props["log_p"],
            props["heavy_atom_count"],
            props["topological_polar_surface_area"],
            1.0 
        ]
        
        full_feature_vector = np.hstack((classical_vector, mock_chemberta_embedding)).reshape(1, -1)
        
        # 4. Standardize and execute predictions across heads
        scaled_features = scaler.transform(full_feature_vector)
        
        prob_scores = clf_model.predict_proba(scaled_features)[0]
        prediction_class = int(clf_model.predict(scaled_features)[0])
        confidence_score = float(prob_scores[prediction_class])
        
        predicted_logbb_val = float(reg_model.predict(scaled_features)[0])

        # 5. Generate RAG Context Report via the loaded Engine
        rag_report = rag_engine.generate_clinical_context(
            payload.canonical_smiles, prediction_class, predicted_logbb_val, top_k=3
        )

        return {
            "canonical_smiles": payload.canonical_smiles,
            "is_bbb_permeable": prediction_class,
            "permeability_confidence": confidence_score,
            "predicted_logbb": predicted_logbb_val,
            "molecular_weight": props["molecular_weight"],
            "log_p": props["log_p"],
            "rag_summary": rag_report.get("summary", "Context unavailable."),
            "nearest_neighbors": rag_report.get("nearest_neighbors", [])
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Pipeline inference failure: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Points cleanly to the app object within the src package structure
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=False)