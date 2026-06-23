import os
import pickle
import asyncio
import numpy as np
import pandas as pd
import chromadb
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from src.embedder import ChemBertaEmbedder
from src.pipeline import ChemblPipelineEngine

# =====================================================================
# INITIALIZE FASTAPI APPLICATION
# =====================================================================
app = FastAPI(
    title="Multi-Modal Neuro-Chemical BBBP RAG Platform",
    description="Production-grade asynchronous microservice incorporating standardized feature transformations.",
    version="1.1.0"
)

# Global memory handles for runtime state optimization
SYSTEM_MODELS = {}
EMBEDDING_ENGINE = None
PIPELINE_ENGINE = None
CHROMA_COLLECTION = None

# =====================================================================
# DEFINE API ROUTING INTERFACE SCHEMAS (Pydantic models)
# =====================================================================
class PermeabilityPredictionRequest(BaseModel):
    """Enforces valid molecular inputs at the API gateway boundary."""
    smiles: str = Field(..., example="COCc1cc(cc2c1CC(C2=O)CC3CCN(CC3)Cc4ccccc4)OC", description="Valid SMILES notation of target drug compound")

class PermeabilityPredictionResponse(BaseModel):
    """Defines type-safe structural schemas for output analytical telemetry."""
    status: str
    target_smiles: str
    predicted_logbb: float
    permeability_class: str
    nearest_historical_matches: list
    azure_cloud_sync: str

# =====================================================================
# WELCOME LANDING ROUTE
# =====================================================================
@app.get("/")
def read_root():
    """Returns a friendly landing confirmation for browser-level traffic status."""
    return {
        "platform": "Multi-Modal Neuro-Chemical BBBP RAG Platform",
        "status": "ONLINE",
        "interactive_docs_url": "http://127.0.0.1:8000/docs"
    }

# =====================================================================
# ON_STARTUP APPLICATION HOOK
# =====================================================================
@app.on_event("startup")
async def initialize_application_infrastructure():
    """Bootstraps models, pre-allocates vectors, and populates local vector stores."""
    global EMBEDDING_ENGINE, PIPELINE_ENGINE, CHROMA_COLLECTION, SYSTEM_MODELS
    
    print("\n🚀 Bootstrapping Standardized Full-Stack Infrastructure...")
    
    scaler_path = "models/fitted_scaler.pkl"
    reg_path = "models/rf_regressor.pkl"
    clf_path = "models/logistic_classifier.pkl"
    csv_path = "data/ingested_bbbp_compounds.csv"
    npy_path = "data/chemberta_embeddings.npy"
    
    if not all(os.path.exists(p) for p in [scaler_path, reg_path, clf_path, csv_path, npy_path]):
        raise RuntimeError("❌ Local pipeline artifacts missing. Please run src/models.py first!")

    # LOAD the pipeline scaler state alongside the predictive models
    with open(scaler_path, "rb") as f:
        SYSTEM_MODELS["scaler"] = pickle.load(f)
    with open(reg_path, "rb") as f:
        SYSTEM_MODELS["regressor"] = pickle.load(f)
    with open(clf_path, "rb") as f:
        SYSTEM_MODELS["classifier"] = pickle.load(f)

    # Initialize processing engines
    EMBEDDING_ENGINE = ChemBertaEmbedder()
    PIPELINE_ENGINE = ChemblPipelineEngine()

    # Initialize Persistent Local Vector Database
    chroma_client = chromadb.PersistentClient(path="data/chroma_db")
    CHROMA_COLLECTION = chroma_client.get_or_create_collection(name="bbbp_chemical_space")

    df_history = pd.read_csv(csv_path)
    embeddings_matrix = np.load(npy_path)

    if CHROMA_COLLECTION.count() == 0:
        print("📥 Populating Vector Database with ChemBERTa embeddings...")
        ids = df_history["molecule_id"].astype(str).tolist()
        smiles_metadata = [{"smiles": s} for s in df_history["canonical_smiles"].tolist()]
        CHROMA_COLLECTION.add(
            embeddings=embeddings_matrix.tolist(),
            metadatas=smiles_metadata,
            ids=ids
        )
        print(f"✅ Indexed {CHROMA_COLLECTION.count()} compounds in ChromaDB.")
    else:
        print(f"ℹ️ Local ChromaDB collection active with {CHROMA_COLLECTION.count()} registered compounds.")

# =====================================================================
# ASYNC TASK: SIMULATE AZURE CLOUD OUTBOUND SHIPMENT
# =====================================================================
async def simulate_azure_cloud_storage_sync(molecule_id: str, payload: dict):
    """Simulates background telemetry export via the azure-ai-ml SDK architecture."""
    await asyncio.sleep(0.1)
    print(f"☁️ [Azure Cloud Logging] Successfully backed up prediction block for '{molecule_id}' to cloud container.")

# =====================================================================
# POST ROUTE: RUN MULTI-MODAL PREDICTION PIPELINE
# =====================================================================
@app.post("/predict", response_model=PermeabilityPredictionResponse)
async def predict_barrier_penetration(request: PermeabilityPredictionRequest, background_tasks: BackgroundTasks):
    """Executes feature normalization and multi-head prediction inside an async wrapper."""
    global EMBEDDING_ENGINE, PIPELINE_ENGINE, CHROMA_COLLECTION, SYSTEM_MODELS

    # 1. Generate deep learning vector coordinate embeddings
    live_vector = EMBEDDING_ENGINE.generate_smiles_embedding(request.smiles)
    if live_vector is None:
        raise HTTPException(status_code=400, detail="Provided chemical string could not be tokenized.")

    # 2. Extract classical chemistry descriptors
    classical_features_obj = PIPELINE_ENGINE.extract_cheminformatics_features(molecule_id="LIVE_QUERY", smiles=request.smiles)
    if classical_features_obj is None:
        raise HTTPException(status_code=400, detail="Cheminformatics metrics calculation failed.")

    # 3. Pull structural nearest neighbors via RAG Database Lookups
    rag_results = CHROMA_COLLECTION.query(
        query_embeddings=[live_vector.tolist()],
        n_results=2
    )
    nearest_neighbors = rag_results.get("ids", [[]])[0]

    # 4. Construct unified raw design matrix row
    flattened_classical = np.array(classical_features_obj.to_feature_list())
    raw_feature_vector = np.hstack((flattened_classical, live_vector)).reshape(1, -1)

    # 5. TRANSFORM input array using the loaded training scaler parameters
    scaled_feature_vector = SYSTEM_MODELS["scaler"].transform(raw_feature_vector)

    # 6. Execute model inference using the normalized data profiles
    predicted_logbb = float(SYSTEM_MODELS["regressor"].predict(scaled_feature_vector)[0])
    predicted_class_id = int(SYSTEM_MODELS["classifier"].predict(scaled_feature_vector)[0])
    class_label = "BBB Permeable (BBB+)" if predicted_class_id == 1 else "Blocked (BBB-)"

    # 7. Route transaction records to background logging queue
    log_id = f"TX_{np.random.randint(10000, 99999)}"
    background_tasks.add_task(
        simulate_azure_cloud_storage_sync,
        molecule_id=log_id,
        payload={"smiles": request.smiles, "logBB": predicted_logbb, "class": class_label}
    )

    return PermeabilityPredictionResponse(
        status="SUCCESS",
        target_smiles=request.smiles,
        predicted_logbb=round(predicted_logbb, 4),
        permeability_class=class_label,
        nearest_historical_matches=nearest_neighbors,
        azure_cloud_sync="COMMITTED_IN_BACKGROUND"
    )