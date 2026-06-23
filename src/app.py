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
    description="Asynchronous microservice wrapping ChemBERTa vector spaces and classical ML heads.",
    version="1.0.0"
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
    smiles: str = Field(..., example="CC(=O)NC1=CC=C(O)C=C1", description="Valid SMILES notation of target drug compound")

class PermeabilityPredictionResponse(BaseModel):
    """Defines type-safe structural schemas for output analytical telemetry."""
    status: str
    target_smiles: str
    predicted_logbb: float
    permeability_class: str
    nearest_historical_matches: list
    azure_cloud_sync: str

# =====================================================================
# ON_STARTUP APPLICATION HOOK
# =====================================================================
@app.on_event("startup")
async def initialize_application_infrastructure():
    """Bootstraps models, pre-allocates vectors, and populates local vector stores."""
    global EMBEDDING_ENGINE, PIPELINE_ENGINE, CHROMA_COLLECTION, SYSTEM_MODELS
    
    print("\n🚀 Bootstrapping Full-Stack Enterprise Workspaces...")
    
    # Paths checking
    reg_path = "models/rf_regressor.pkl"
    clf_path = "models/logistic_classifier.pkl"
    csv_path = "data/ingested_bbbp_compounds.csv"
    npy_path = "data/chemberta_embeddings.npy"
    
    if not all(os.path.exists(p) for p in [reg_path, clf_path, csv_path, npy_path]):
        raise RuntimeError("❌ Local training assets missing. Run pipeline, embedder, and models files first!")

    # Load estimators into global tracking dictionary
    with open(reg_path, "rb") as f:
        SYSTEM_MODELS["regressor"] = pickle.load(f)
    with open(clf_path, "rb") as f:
        SYSTEM_MODELS["classifier"] = pickle.load(f)

    # Initialize shared pipeline engines
    EMBEDDING_ENGINE = ChemBertaEmbedder()
    PIPELINE_ENGINE = ChemblPipelineEngine()

    # INITIALIZE Persistent Local ChromaDB Client
    chroma_client = chromadb.PersistentClient(path="data/chroma_db")
    # CREATE OR GET ChromaDB Collection named "bbbp_chemical_space"
    CHROMA_COLLECTION = chroma_client.get_or_create_collection(name="bbbp_chemical_space")

    # POPULATE local ChromaDB Collection with historic embeddings for RAG lookups
    df_history = pd.read_csv(csv_path)
    embeddings_matrix = np.load(npy_path)

    # Ingest historical matrix layers into vector memory space if empty
    if CHROMA_COLLECTION.count() == 0:
        print("📥 Vector Database empty. Indexing historical ChemBERTa embeddings for RAG lookups...")
        ids = df_history["molecule_id"].astype(str).tolist()
        smiles_metadata = [{"smiles": s} for s in df_history["canonical_smiles"].tolist()]
        embeddings_list = embeddings_matrix.tolist()

        CHROMA_COLLECTION.add(
            embeddings=embeddings_list,
            metadatas=smiles_metadata,
            ids=ids
        )
        print(f"✅ Successfully indexed {CHROMA_COLLECTION.count()} historical reference compounds in ChromaDB.")
    else:
        print(f"ℹ️ Local ChromaDB collection active with {CHROMA_COLLECTION.count()} registered compounds.")

# =====================================================================
# ASYNC FUNCTION mock_azure_blob_sync(payload)
# =====================================================================
async def simulate_azure_cloud_storage_sync(molecule_id: str, payload: dict):
    """Simulates background execution mapping telemetry records to cloud blobs via azure-ai-ml."""
    # Simulates cloud networking handshakes without blocking our core API response thread
    await asyncio.sleep(0.1)
    print(f"☁️ [Azure Cloud Logging] Successfully backed up prediction block for '{molecule_id}' to container: 'sys-telemetry-logs'.")

# =====================================================================
# ASYNC FUNCTION predict_barrier_penetration(Request)
# =====================================================================
@app.post("/predict", response_model=PermeabilityPredictionResponse)
async def predict_barrier_penetration(request: PermeabilityPredictionRequest, background_tasks: BackgroundTasks):
    """
    Executes live RAG semantic nearest-neighbor matching, processes RDKit structural features,
    and runs multi-head statistical inference inside an asynchronous gateway wrapper.
    """
    global EMBEDDING_ENGINE, PIPELINE_ENGINE, CHROMA_COLLECTION, SYSTEM_MODELS

    # GENERATE on-the-fly ChemBERTa structural embedding vector
    live_vector = EMBEDDING_ENGINE.generate_smiles_embedding(request.smiles)
    if live_vector is None:
        raise HTTPException(status_code=400, detail="Provided chemical string could not be compiled by RDKit or ChemBERTa.")

    # GENERATE classical structural properties using RDKit pipeline matrix
    classical_features_obj = PIPELINE_ENGINE.extract_cheminformatics_features(molecule_id="LIVE_QUERY", smiles=request.smiles)
    if classical_features_obj is None:
        raise HTTPException(status_code=400, detail="Structural feature calculation constraints failed on input SMILES.")

    # QUERY ChromaDB Collection for top 2 nearest neighbor molecule matches
    rag_results = CHROMA_COLLECTION.query(
        query_embeddings=[live_vector.tolist()],
        n_results=2
    )
    
    nearest_neighbors = rag_results.get("ids", [[]])[0]

    # CONCATENATE live properties + embeddings horizontally to match model signature arrays
    flattened_classical = np.array(classical_features_obj.to_feature_list())
    full_feature_vector = np.hstack((flattened_classical, live_vector)).reshape(1, -1)

    # EXECUTE Random Forest Regressor calculation for logBB
    predicted_logbb = float(SYSTEM_MODELS["regressor"].predict(full_feature_vector)[0])
    
    # EXECUTE Logistic Regression Classifier calculation for binary label
    predicted_class_id = int(SYSTEM_MODELS["classifier"].predict(full_feature_vector)[0])
    class_label = "BBB Permeable (BBB+)" if predicted_class_id == 1 else "Blocked (BBB-)"

    # TRIGGER async background task to simulate Azure Cloud storage archival sync ($0 total costs)
    log_id = f"TX_{np.random.randint(10000, 99999)}"
    background_tasks.add_task(
        simulate_azure_cloud_storage_sync,
        molecule_id=log_id,
        payload={"smiles": request.smiles, "logBB": predicted_logbb, "class": class_label}
    )

    # COMPILE response payload matching Prediction Response Schema
    return PermeabilityPredictionResponse(
        status="SUCCESS",
        target_smiles=request.smiles,
        predicted_logbb=round(predicted_logbb, 4),
        permeability_class=class_label,
        nearest_historical_matches=nearest_neighbors,
        azure_cloud_sync="COMMITTED_IN_BACKGROUND"
    )