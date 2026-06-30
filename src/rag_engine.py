# src/rag_engine.py
import numpy as np
from src.vector_store import MolecularVectorStore

class MolecularRAGEngine:
    def __init__(self, db_path: str = "data/chroma_db"):
        """Initializes the RAG retrieval router using our persistent ChromaDB instance."""
        self.v_store = MolecularVectorStore(db_path=db_path)

    def retrieve_molecular_context(self, query_embedding: np.ndarray, n_results: int = 3) -> str:
        """Finds closest neighbors and formats them into a clean text prompt context block."""
        # Convert matrix back to flat list for ChromaDB API compatibility
        flat_embedding = query_embedding.flatten().tolist()
        
        # Execute Vector Space Search
        raw_results = self.v_store.query_nearest_neighbors(flat_embedding, n_results=n_results)
        
        if not raw_results or not raw_results['documents'] or len(raw_results['documents'][0]) == 0:
            return "No historical molecular analogs found in the current vector space knowledge base."

        # Compile matching records into a structured prompt layout
        context_blocks = []
        for i in range(len(raw_results['documents'][0])):
            smiles = raw_results['documents'][0][i]
            meta = raw_results['metadatas'][0][i]
            distance = raw_results['distances'][0][i] if 'distances' in raw_results else 0.0
            
            block = (
                f"--- Reference Compound Analog #{i+1} ---\n"
                f"• Molecule ID: {meta.get('compound_name', 'Unknown')}\n"
                f"• Structural Key (SMILES): {smiles}\n"
                f"• Calculated LogP: {meta.get('logBB', 'N/A')}\n"
                f"• Experimental BBB Permeability State: {meta.get('p_np', 'N/A')}\n"
                f"• Vector Cosine Spatial Distance: {distance:.4f}\n"
            )
            context_blocks.append(block)

        return "\n".join(context_blocks)

if __name__ == "__main__":
    # Smoke test to make sure retrieval routes execute perfectly
    print("🧪 Testing RAG Routing Channel...")
    engine = MolecularRAGEngine()
    
    # Create a mock 768-D query dimension vector
    mock_vector = np.random.uniform(-1, 1, (1, 768))
    retrieved_text = engine.retrieve_molecular_context(mock_vector, n_results=2)
    print("\n📬 Sample Retrieval Output Context Window:\n")
    print(retrieved_text)