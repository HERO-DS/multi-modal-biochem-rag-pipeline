# src/vector_store.py
import os
import numpy as np
import pandas as pd
import chromadb

class MolecularVectorStore:
    def __init__(self, db_path: str = "data/chroma_db"):
        """Initializes persistent disk storage for molecular embeddings."""
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Initialize or fetch our molecular collection using Cosine Distance
        self.collection = self.client.get_or_create_collection(
            name="chemberta_molecules",
            metadata={"hnsw:space": "cosine"}
        )

    def populate_database(self, dataset_path: str, embeddings_path: str):
        """Loads physical molecule data profiles and pushes them to ChromaDB."""
        if not os.path.exists(dataset_path) or not os.path.exists(embeddings_path):
            raise FileNotFoundError("❌ Missing base compound file records or matrix array.")

        # Read historical artifacts
        df_compounds = pd.read_csv(dataset_path)
        embeddings = np.load(embeddings_path)
        
        print(f"📦 Found {len(df_compounds)} structures inside dataset index. Seeding Vector Database...")

        ids = []
        documents = []
        metadatas = []
        embeddings_list = []

        for idx, row in df_compounds.iterrows():
            if idx >= len(embeddings):
                break  # Protect against shape alignment boundaries
                
            mol_id = f"mol_{idx}"
            ids.append(mol_id)
            
            # Map canonical_smiles directly as the document text
            documents.append(str(row['canonical_smiles']))
            
            # Pull your exact structural labels for the RAG context metadata
            p_np_val = row.get('experimental_permeability', 'Unknown')
            logBB_val = row.get('log_p', 0.0)  # Using log_p as our continuous target feature fallback
            
            metadatas.append({
                "p_np": str(p_np_val),
                "logBB": float(logBB_val) if pd.notna(logBB_val) else 0.0,
                "compound_name": str(row['molecule_id'])
            })
            
            embeddings_list.append(embeddings[idx].tolist())

        # Sync data payloads into local database collection clusters
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents
        )
        print(f"✅ Vector database successfully updated! Total indexed entries: {self.collection.count()}")

    def query_nearest_neighbors(self, query_embedding: list, n_results: int = 3) -> dict:
        """Searches vector space for the top N structurally similar molecules."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results

if __name__ == "__main__":
    v_store = MolecularVectorStore()
    v_store.populate_database(
        dataset_path="data/ingested_bbbp_compounds.csv",
        embeddings_path="data/chemberta_embeddings.npy"
    )