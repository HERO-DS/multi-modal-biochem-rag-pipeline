# src/rag_engine.py
import os
import numpy as np
import google.generativeai as genai
from src.vector_store import MolecularVectorStore

class MolecularRAGEngine:
    def __init__(self, db_path: str = "chroma_db"):
        """Initializes the RAG retrieval router and the LLM generation context."""
        # Fixed path to point directly to your root chroma_db folder
        self.v_store = MolecularVectorStore(db_path=db_path)
        
        # Configure the free Gemini API key (safely pulled from system environment)
        api_key = os.getenv("GEMINI_API_KEY", "")
        if api_key:
            genai.configure(api_key=api_key)
            # Updated to the active Gemini 2.5 production engine
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.model = None

    def retrieve_molecular_context(self, query_embedding: np.ndarray, n_results: int = 3) -> str:
        """Finds closest neighbors and formats them into a clean text context block."""
        flat_embedding = query_embedding.flatten().tolist()
        raw_results = self.v_store.query_nearest_neighbors(flat_embedding, n_results=n_results)
        
        if not raw_results or not raw_results['documents'] or len(raw_results['documents'][0]) == 0:
            return "No historical molecular analogs found in the current vector space knowledge base."

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
                f"• Experimental BBB Permeability State: {'Permeable (1)' if str(meta.get('p_np')) == '1' else 'Non-Permeable (0)'}\n"
                f"• Spatial Distance: {distance:.4f}\n"
            )
            context_blocks.append(block)

        return "\n".join(context_blocks)

    def generate_biochem_report(self, smiles: str, logBB: float, conf: float, is_permeable: bool, context: str) -> str:
        """Uses the cloud LLM to synthesize data points into a professional research report."""
        if not self.model:
            return (
                "⚠️ LLM Report Generation Offline: 'GEMINI_API_KEY' environment variable not detected.\n"
                "Please set your API key to unlock automated expert chemical dossier summaries."
            )

        prompt = f"""
        You are an expert computational toxicologist and medicinal chemist. Analyze the following target molecule and synthesize a brief, structured scientific report.

        TARGET MOLECULE:
        - SMILES Notation: {smiles}
        
        PREDICTED MACHINE LEARNING PROPERTIES:
        - Estimated LogBB (Partition Coefficient): {logBB:.4f}
        - Blood-Brain Barrier (BBB) Permeability Class: {'BBB+ (Crosses Barrier)' if is_permeable else 'BBB- (Blocked)'}
        - Model Confidence: {conf*100:.2f}%

        HISTORICAL RETRIEVED NEIGHBORS (VECTOR SPACE ANALOGS):
        {context}

        INSTRUCTIONS:
        Write a professional 3-section evaluation details report:
        1. MOLECULAR EVALUATION: Analyze the SMILES structural features and properties.
        2. RAG ANALOG REFLECTION: Contrast this target molecule with the retrieved database analogs.
        3. CLINICAL RECOMMENDATION: Provide a conclusion on its potential for central nervous system (CNS) targeting or avoidance.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Gemini API Error: {str(e)}"