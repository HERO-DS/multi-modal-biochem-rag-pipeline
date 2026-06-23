import os
import torch
import numpy as np
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel

# =====================================================================
# DEFINE Embedding Output Schema (Pydantic Model)
# =====================================================================
class MolecularEmbeddingVector(BaseModel):
    """Validates the structural integrity and dimensionality of generated embedding spaces."""
    molecule_id: str = Field(..., description="Unique compound identifier string matching the ingestion tier")
    embedding: List[float] = Field(..., description="High-dimensional continuous coordinate feature vector")

# =====================================================================
# CLASS ChemBertaEmbedder
# =====================================================================
class ChemBertaEmbedder:
    """Handles deep-learning tokenization and vector space feature mapping via local hardware."""
    
    def __init__(self, model_name: str = "seyonec/ChemBERTa-zinc-base-v1"):
        # DETECT local GPU hardware availability (CUDA vs CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Convert the device object to a string first, then make it uppercase
        print(f"🖥️ Execution Hardware Context Initialized: Using Target Device [{str(self.device).upper()}]")    

        # LOAD pre-trained 'seyonec/ChemBERTa-zinc-base-v1' Tokenizer and Model
        print(f"📥 Loading Hugging Face weights for model blueprint: '{model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # MOVE Model weights to selected hardware device
        self.model.to(self.device)
        # Put model in evaluation mode to turn off dropout/batch normalization steps
        self.model.eval()
        print("✅ Deep Learning Model weights locked into memory space successfully.")

    # =====================================================================
    # FUNCTION generate_smiles_embedding(smiles_string)
    # =====================================================================
    def generate_smiles_embedding(self, smiles: str) -> Optional[np.ndarray]:
        """
        Tokenizes SMILES sequences and extracts pooled hidden state embeddings
        using structural representation rules from the pre-trained transformer model.
        """
        try:
            # TOKENIZE smiles_string to tensor map with padding and truncation
            inputs = self.tokenizer(
                smiles,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )
            
            # MOVE input token tensors to active hardware device
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            # WITH torch.no_grad() context enabled to bypass backpropagation memory overhead
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # EXTRACT last hidden states matrix
                # Shape: [batch_size, sequence_length, hidden_dimension]
                last_hidden_state = outputs.last_hidden_state
                
                # COMPUTE mean pooling across the token sequence dimension to create a uniform structure
                mean_pooled = torch.mean(last_hidden_state, dim=1)
                
                # CONVERT vector tensor back to local host CPU memory as a NumPy array
                embedding_vector = mean_pooled.squeeze().cpu().numpy()
                
            return embedding_vector
            
        except Exception as error:
            print(f" [Embedding Failure] Error generating vector representation for string '{smiles}': {error}")
            return None

    # =====================================================================
    # FUNCTION run_extraction_pipeline(input_csv_path, output_npy_path)
    # =====================================================================
    def run_extraction_pipeline(self, input_csv: str = "data/ingested_bbbp_compounds.csv", output_dir: str = "data") -> np.ndarray:
        """
        Iterates over the ingested dataset, maps molecules into numerical vector spaces,
        and saves output structures as lightning-fast binary matrices for the downstream heads.
        """
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"❌ Target data file not found at: {input_csv}. Please run pipeline.py first.")

        # LOAD data matrix from input_csv_path
        df = pd.read_csv(input_csv)
        print(f"📋 Loaded {len(df)} compound records from ingestion file layer.")

        # INITIALIZE empty storage arrays for vector space and tracking IDs
        embeddings_list = []
        tracking_ids = []

        # FOR each row in compound dataset
        for idx, row in df.iterrows():
            mol_id = str(row["molecule_id"])
            smiles_str = str(row["canonical_smiles"])
            
            # CALL generate_smiles_embedding(canonical_smiles)
            vector = self.generate_smiles_embedding(smiles_str)
            
            # IF vector extraction succeeds, validate structure type and add to final matrix
            if vector is not None:
                try:
                    # Enforce strict typing via our Pydantic validation layout
                    validated_data = MolecularEmbeddingVector(molecule_id=mol_id, embedding=vector.tolist())
                    
                    embeddings_list.append(vector)
                    tracking_ids.append(mol_id)
                except ValidationError as type_error:
                    print(f" [Type Error] Row {idx} failed structural output verification: {type_error}")
                    continue

        # CONVERT memory array into a structured NumPy/Pandas transformation matrix
        feature_matrix = np.array(embeddings_list)
        
        # SAVE final numerical matrix stack as a binary NumPy file (.npy)
        matrix_output_path = os.path.join(output_dir, "chemberta_embeddings.npy")
        np.save(matrix_output_path, feature_matrix)
        
        # SAVE alignment tracking index as a clean CSV reference mapping file
        tracking_df = pd.DataFrame({"molecule_id": tracking_ids})
        tracking_output_path = os.path.join(output_dir, "embeddings_manifest.csv")
        tracking_df.to_csv(tracking_output_path, index=False)
        
        print(f" Finished Feature Extraction Matrix Block!")
        print(f"   -> Numerical Matrix Saved to: {matrix_output_path} (Shape: {feature_matrix.shape})")
        print(f"   -> Validation Index Manifest Saved to: {tracking_output_path}")
        
        return feature_matrix

if __name__ == "__main__":
    # Standard local process execution block to verify deep learning pipeline module
    embedder = ChemBertaEmbedder()
    matrix = embedder.run_extraction_pipeline()
    print("\n--- Diagnostic Embedding Matrix Dimension Signature ---")
    print(f"Matrix Rank Dimensionality: {matrix.ndim}")
    print(f"Sample Top Vector Extract (First 5 Dimensions of Molecule 0): {matrix[0][:5]}")