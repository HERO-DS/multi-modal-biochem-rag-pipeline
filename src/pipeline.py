import os
import pandas as pd
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import Descriptors

# =====================================================================
# TYPE-SAFE DATA SCHEMA (Pydantic model)
# =====================================================================
class MolecularPropertyProfile(BaseModel):
    """Enforces rigorous schema formatting rules for extracted clinical features."""
    molecule_id: str
    canonical_smiles: str
    molecular_weight: float
    log_p: float
    hydrogen_bond_donors: int
    hydrogen_bond_acceptors: int
    topological_polar_surface_area: float
    experimental_permeability: int  # Real clinical benchmark target label

    def to_feature_list(self) -> list:
        """Flattens classical descriptors into an ordered row array for ML heads."""
        return [
            self.molecular_weight,
            self.log_p,
            self.hydrogen_bond_donors,
            self.hydrogen_bond_acceptors,
            self.topological_polar_surface_area
        ]

# =====================================================================
# CLASS ChemblPipelineEngine (Upgraded to MoleculeNet Benchmark)
# =====================================================================
class ChemblPipelineEngine:
    """Automates secure data fetching, feature extraction, and schema enforcement loops."""
    
    def __init__(self, data_directory: str = "data"):
        self.data_dir = data_directory
        os.makedirs(self.data_dir, exist_ok=True)

    def download_benchmark_dataset(self) -> str:
        """Streams the official Stanford MoleculeNet BBBP dataset directly into local storage."""
        local_path = os.path.join(self.data_dir, "raw_bbbp_benchmark.csv")
        
        if os.path.exists(local_path):
            print(f"ℹ️ Found cached MoleculeNet benchmark file locally at: {local_path}")
            return local_path
            
        print("🌐 Connection Active. Streaming Stanford MoleculeNet BBBP Dataset from deep learning repository...")
        # Direct public deep learning benchmark repository URL
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
        df = pd.read_csv(url)
        
        # Save a local cache backup copy to disk
        df.to_csv(local_path, index=False)
        print(f"📥 Download complete. Saved raw reference matrix to: {local_path}")
        return local_path

    def extract_cheminformatics_features(self, molecule_id: str, smiles: str, target_label: int = 0) -> MolecularPropertyProfile:
        """Utilizes local RDKit graph kernels to parse structures into chemical features."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Enforce unified canonical structural formatting
            canonical_smiles = Chem.MolToSmiles(mol)
            
            return MolecularPropertyProfile(
                molecule_id=str(molecule_id),
                canonical_smiles=canonical_smiles,
                molecular_weight=float(Descriptors.MolWt(mol)),
                log_p=float(Descriptors.MolLogP(mol)),
                hydrogen_bond_donors=int(Descriptors.NumHDonors(mol)),
                hydrogen_bond_acceptors=int(Descriptors.NumHAcceptors(mol)),
                topological_polar_surface_area=float(Descriptors.TPSA(mol)),
                experimental_permeability=int(target_label)
            )
        except Exception:
            return None

    def execute_ingestion_pipeline(self, sample_limit: int = 300):
        """Processes raw datasets, validates profiles via Pydantic, and writes features to disk."""
        raw_csv = self.download_benchmark_dataset()
        df_raw = pd.read_csv(raw_csv)
        
        print(f"🧪 Processing and refining chemical nodes. Target Scale Limit: {sample_limit} rows...")
        processed_profiles = []
        
        # Parse records step-by-step
        for idx, row in df_raw.iterrows():
            if len(processed_profiles) >= sample_limit:
                break
                
            # MoleculeNet column mapping: 'name' is ID, 'smiles' is string, 'p_np' is the real experimental label
            smiles_str = str(row["smiles"])
            mol_name = str(row["name"]) if pd.notna(row["name"]) else f"COMP_NUM_{idx}"
            real_target = int(row["p_np"])
            
            profile = self.extract_cheminformatics_features(molecule_id=mol_name, smiles=smiles_str, target_label=real_target)
            if profile is not None:
                processed_profiles.append(profile.dict())

        # Save structured features to disk
        df_output = pd.DataFrame(processed_profiles)
        output_csv_path = os.path.join(self.data_dir, "ingested_bbbp_compounds.csv")
        df_output.to_csv(output_csv_path, index=False)
        
        print(f"✅ Ingestion Engine Completed! Saved structured feature profiles to: {output_csv_path}")

if __name__ == "__main__":
    engine = ChemblPipelineEngine()
    # Let's target 300 clean records for a fast, balanced local training execution
    engine.execute_ingestion_pipeline(sample_limit=300)