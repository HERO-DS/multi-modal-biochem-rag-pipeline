import os
import sys
import pytest
from fastapi.testclient import TestClient

# Anchor parent package context to run pytest from root workspace directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.app import app, compute_classical_rdkit_features

@pytest.fixture
def test_client():
    """Generates an ephemeral mock execution network worker instance."""
    with TestClient(app) as client:
        yield client

def test_molecular_descriptor_calculator():
    """Validates math stability of calculated classical RDKit features."""
    valid_smiles = "CCN(CC)CC"  # Triethylamine
    metrics = compute_classical_rdkit_features(valid_smiles)
    
    assert "molecular_weight" in metrics
    assert "log_p" in metrics
    assert metrics["molecular_weight"] > 0
    
    with pytest.raises(ValueError):
        compute_classical_rdkit_features("INVALID_SMILES_STRING")

def test_prediction_endpoint_integration(test_client):
    """Executes a full integration round-trip across models and the RAG engine."""
    payload = {"canonical_smiles": "CCN(CC)CC"}
    response = test_client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "is_bbb_permeable" in data
    assert "predicted_logbb" in data
    assert "rag_summary" in data
    assert len(data["nearest_neighbors"]) <= 3