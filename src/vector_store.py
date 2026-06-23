import os
import numpy as np
import pandas as pd

class BiochemicalKnowledgeStore:
    """Manages chemical embedding indexing and cross-referencing for RAG context retrieval."""
    
    def __init__(self, data_csv: str = "data/ingested_bbbp_compounds.csv", 
                 embeddings_npy: str = "data/chemberta_embeddings.npy"):
        self.data_csv = data_csv
        self.embeddings_npy = embeddings_npy
        self.metadata = None
        self.embeddings = None
        
    def load_vector_index(self):
        """Builds in-memory semantic indices linking molecules to experimental targets."""
        if not os.path.exists(self.data_csv) or not os.path.exists(self.embeddings_npy):
            raise FileNotFoundError("Pipeline files missing. Ensure preprocessing stages are finished.")
            
        self.metadata = pd.read_csv(self.data_csv)
        self.embeddings = np.load(self.embeddings_npy)
        print(f"📦 Loaded vector search index with {self.embeddings.shape[0]} tracking nodes.")

    def query_nearest_compounds(self, query_embedding: np.ndarray, top_k: int = 3) -> pd.DataFrame:
        """Executes a vector search query against the index database."""
        if self.embeddings is None or self.metadata is None:
            self.load_vector_index()
            
        # Compute standard L2 Euclidean distance across vector axes
        distances = np.linalg.norm(self.embeddings - query_embedding, axis=1)
        nearest_indices = np.argsort(distances)[:top_k]
        
        matched_metadata = self.metadata.iloc[nearest_indices].copy()
        matched_metadata["similarity_distance_score"] = distances[nearest_indices]
        return matched_metadata

if __name__ == "__main__":
    store = BiochemicalKnowledgeStore()
    store.load_vector_index()
    # Test vector resolution matching an arbitrary mock vector query
    sample_query = np.random.normal(0.0, 0.1, 768)
    hits = store.query_nearest_compounds(sample_query, top_k=2)
    print("\n🔍 Test Hit Metadata Results:\n", hits[["molecule_id", "canonical_smiles", "similarity_distance_score"]])