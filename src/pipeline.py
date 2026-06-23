import warnings
# Silence the deprecated pkg_resources warning bubbling up from the third-party chembl client
warnings.filterwarnings("ignore", category=UserWarning, module="chembl_webresource_client")

import os
import asyncio
import numpy as np
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError
from rdkit import Chem
from rdkit.Chem import Descriptors

# =====================================================================
# SAFE IMPORT: Wrap the eager ChEMBL client inside an import shield
# =====================================================================
try:
    from chembl_webresource_client.new_client import new_client
    CHEMBL_AVAILABLE = True
except Exception:
    print("⚠️ ChEMBL library failed to initialize due to upstream server downtime. Isolation protocols active.")
    new_client = None
    CHEMBL_AVAILABLE = False

# =====================================================================
# DEFINE Type-Safe Chemical Record Schema (Pydantic model)
# =====================================================================
class RawChemicalRecord(BaseModel):
    """Validates raw input data streams coming from ChEMBL or secondary benchmarks."""
    molecule_chembl_id: str = Field(..., description="Unique Identification Token")
    canonical_smiles: str = Field(..., description="SMILES string representation of 2D molecular structure")

# =====================================================================
# DEFINE Target Feature Output Schema (Pydantic model)
# =====================================================================
class ProcessedChemicalFeatures(BaseModel):
    """Enforces strict structural typing for features feeding the ML transformation matrix."""
    molecule_id: str
    canonical_smiles: str
    molecular_weight: float
    logp: float
    tpsa: float
    h_bond_donors: int
    h_bond_acceptors: int

    def to_feature_list(self) -> List[float]:
        """Flattens numerical values directly into an ordered vector space."""
        return [
            self.molecular_weight,
            self.logp,
            self.tpsa,
            self.h_bond_donors,
            self.h_bond_acceptors
        ]

# =====================================================================
# INITIALIZE Pipeline Engine
# =====================================================================
class ChemblPipelineEngine:
    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

# =====================================================================
    # ASYNC FUNCTION fetch_chembl_compounds(limit)
    # =====================================================================
    async def fetch_chembl_compounds(self, limit: int = 50) -> List[dict]:
        """
        Executes synchronous ChEMBL queries safely if the library initialized.
        """
        if not CHEMBL_AVAILABLE or new_client is None:
            raise ConnectionError("ChEMBL remote client environment is marked offline.")

        loop = asyncio.get_running_loop()
        
        def _execute_query():
            molecule_query = new_client.molecule
            # Updated the filter to use 'full_mwt' instead of 'mw' based on server diagnostics
            records = molecule_query.filter(
                molecule_properties__full_mwt__lte=500
            ).only(['molecule_chembl_id', 'molecule_structures'])[0:limit]
            
            cleaned_records = []
            for record in records:
                structures = record.get("molecule_structures")
                if structures and "canonical_smiles" in structures:
                    cleaned_records.append({
                        "molecule_chembl_id": record["molecule_chembl_id"],
                        "canonical_smiles": structures["canonical_smiles"]
                    })
            return cleaned_records

        return await loop.run_in_executor(None, _execute_query)

    # =====================================================================
    # FALLBACK FUNCTION: fetch_fallback_bbbp_dataset(limit)
    # =====================================================================
    def fetch_fallback_bbbp_dataset(self, limit: int = 50) -> List[dict]:
        """
        Fallback stream downloading classic Moleculenet BBBP benchmark data
        if the primary ChEMBL server is fully unreachable.
        """
        print("🔄 Rerouting Data Stream: Accessing alternative stable BBBP dataset mirror...")
        fallback_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
        
        # Read directly from public CSV cache mirror
        df = pd.read_csv(fallback_url, nrows=limit)
        
        cleaned_records = []
        for idx, row in df.iterrows():
            if pd.notna(row['smiles']):
                cleaned_records.append({
                    "molecule_chembl_id": f"BENCHMARK_{row['num']}",
                    "canonical_smiles": row['smiles']
                })
        return cleaned_records

    # =====================================================================
    # FUNCTION extract_cheminformatics_features(smiles_string)
    # =====================================================================
    def extract_cheminformatics_features(self, molecule_id: str, smiles: str) -> Optional[ProcessedChemicalFeatures]:
        """
        Parses SMILES strings using RDKit topological rule matrices.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            mw = float(Descriptors.MolWt(mol))
            logp = float(Descriptors.MolLogP(mol))
            tpsa = float(Descriptors.TPSA(mol))
            h_donors = int(Descriptors.NumHDonors(mol))
            h_acceptors = int(Descriptors.NumHAcceptors(mol))
            
            return ProcessedChemicalFeatures(
                molecule_id=molecule_id,
                canonical_smiles=smiles,
                molecular_weight=mw,
                logp=logp,
                tpsa=tpsa,
                h_bond_donors=h_donors,
                h_bond_acceptors=h_acceptors
            )
        except Exception as error:
            print(f" [Exception] Runtime error parsing features for {molecule_id}: {error}")
            return None

    # =====================================================================
    # ASYNC FUNCTION run_ingestion_pipeline()
    # =====================================================================
    async def run_ingestion_pipeline(self, record_limit: int = 50) -> pd.DataFrame:
        """
        Runs full loop workflow with advanced import-level fault tolerance.
        """
        print(f"⚡ Launching Pipeline Engine. Targeting {record_limit} compounds...")
        
        # Attempt primary live connection; instantly catch failures or load-time errors
        try:
            raw_payloads = await self.fetch_chembl_compounds(limit=record_limit)
        except Exception as error:
            print(f"❌ Primary live endpoint unreachable ({error}). Triggering safe fallbacks.")
            raw_payloads = self.fetch_fallback_bbbp_dataset(limit=record_limit)
        
        processed_dataset: List[ProcessedChemicalFeatures] = []
        
        for raw_data in raw_payloads:
            try:
                validated_input = RawChemicalRecord(**raw_data)
                features = self.extract_cheminformatics_features(
                    molecule_id=validated_input.molecule_chembl_id,
                    smiles=validated_input.canonical_smiles
                )
                if features:
                    processed_dataset.append(features)
            except ValidationError:
                continue

        if not processed_dataset:
            print("❌ Pipeline finished with zero valid compound extractions.")
            return pd.DataFrame()
            
        data_matrix = []
        for item in processed_dataset:
            row = {
                "molecule_id": item.molecule_id,
                "canonical_smiles": item.canonical_smiles,
                "molecular_weight": item.molecular_weight,
                "logp": item.logp,
                "tpsa": item.tpsa,
                "h_bond_donors": item.h_bond_donors,
                "h_bond_acceptors": item.h_bond_acceptors
            }
            data_matrix.append(row)
            
        df_matrix = pd.DataFrame(data_matrix)
        
        output_path = os.path.join(self.output_dir, "ingested_bbbp_compounds.csv")
        df_matrix.to_csv(output_path, index=False)
        print(f" Matrix Transformation Success! Saved {len(df_matrix)} rows to: {output_path}")
        
        return df_matrix

if __name__ == "__main__":
    engine = ChemblPipelineEngine()
    transformed_matrix = asyncio.run(engine.run_ingestion_pipeline(record_limit=200))
    print("\n--- Diagnostic Pipeline Matrix Head Preview ---")
    print(transformed_matrix.head(3))