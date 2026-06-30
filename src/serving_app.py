# src/serving_app.py
import os
import pickle
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# Import your embedder using clean relative-path logic
from src.embedder import ChemBertaEmbedder

# 1. Request / Response schemas
class SmilesPayload(BaseModel):
    smiles: str = Field(..., json_schema_extra={"example": "CCO"}, description="Raw SMILES string of the molecule")

class InferenceResponse(BaseModel):
    smiles: str
    logBB_prediction: float
    is_permeable: bool
    permeability_probability: float
    embedding_sample: List[float] = Field(..., description="First 5 dimensions of the 768-D vector")

# 2. Initialize FastAPI App
app = FastAPI(
    title="Biochem RAG Asynchronous Serving Layer",
    description="Production API engine for molecular properties and vector extractions.",
    version="1.0.0"
)

# Global model containers
embedder = None
rf_regressor = None
logistic_classifier = None

@app.on_event("startup")
def load_assets():
    """Loads weights and pickle models from their dedicated folders on startup."""
    global embedder, rf_regressor, logistic_classifier
    print("⏳ Initializing production model assets in memory...")
    
    embedder = ChemBertaEmbedder()
    
    # Pointing exactly to your correct models directory
    rf_path = os.path.join("models", "rf_regressor.pkl")
    logistic_path = os.path.join("models", "logistic_classifier.pkl")
    
    if not os.path.exists(rf_path) or not os.path.exists(logistic_path):
        raise FileNotFoundError(f"❌ Pickled ML models not found! Expected at {rf_path} and {logistic_path}")
        
    with open(rf_path, "rb") as f:
        rf_regressor = pickle.load(f)
    with open(logistic_path, "rb") as f:
        logistic_classifier = pickle.load(f)
        
    print("🚀 API Serving Layer Standby: Ready for Async Inference requests.")

# 3. Asynchronous Endpoint
@app.post("/api/v1/predict", response_model=InferenceResponse)
async def predict_molecule(payload: SmilesPayload):
    if not payload.smiles.strip():
        raise HTTPException(status_code=400, detail="SMILES string cannot be empty.")
        
    try:
        # Extract 768-D vector using the frozen model weights
        with torch.no_grad():
            raw_embedding = embedder.generate_smiles_embedding(payload.smiles)
            
        embedding_array = np.array(raw_embedding).reshape(1, -1)
        
        # Match your design space (15, 773) by padding the 5 hand-crafted features
        feature_padding = np.zeros((1, 6)) 
        full_design_space = np.hstack([embedding_array, feature_padding])
        
        # Run inference via ML heads
        logBB_pred = float(rf_regressor.predict(full_design_space)[0])
        prob_pred = float(logistic_classifier.predict_proba(full_design_space)[0][1])
        class_pred = bool(logistic_classifier.predict(full_design_space)[0] == "BBB+")

        return InferenceResponse(
            smiles=payload.smiles,
            logBB_prediction=logBB_pred,
            is_permeable=class_pred,
            permeability_probability=prob_pred,
            embedding_sample=embedding_array[0][:5].tolist()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Pipeline Error: {str(e)}")