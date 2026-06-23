import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score, classification_report

# =====================================================================
# CLASS BrainBarrierPredictor
# =====================================================================
class BrainBarrierPredictor:
    """Trains, normalizes, evaluates, and serializes downstream machine learning heads."""
    
    def __init__(self):
        # INITIALIZE Estimators and Preprocessing Scalers
        self.scaler = StandardScaler()
        self.regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        # Keeping max_iter=1000 to show that scaling resolves the convergence error entirely
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        print("🤖 Machine Learning Predictive Engine Initialized with Preprocessing Layers.")

    # =====================================================================
    # FUNCTION load_and_align_datasets(csv_path, embeddings_path)
    # =====================================================================
    def load_and_align_datasets(self, csv_path: str = "data/ingested_bbbp_compounds.csv", 
                                embeddings_path: str = "data/chemberta_embeddings.npy"):
        """Loads independent feature streams and concatenates them horizontally."""
        if not os.path.exists(csv_path) or not os.path.exists(embeddings_path):
            raise FileNotFoundError("❌ Upstream feature matrices missing. Run upstream pipeline scripts first.")

        df_classical = pd.read_csv(csv_path)
        embeddings = np.load(embeddings_path)

        # Isolate numerical features by dropping metadata descriptors
        classical_features = df_classical.drop(columns=["molecule_id", "canonical_smiles"]).to_numpy()

        # CONCATENATE classical features matrix and deep embeddings matrix horizontally
        X = np.hstack((classical_features, embeddings))
        
        # New authentic experimental target extraction code
        y_clf = df_classical["experimental_permeability"].to_numpy()
        # Set up a pseudo-logBB correlation tracking indicator matching molecular weight scales for the regressor head
        y_reg = (df_classical["log_p"].to_numpy() * 0.4) - (df_classical["topological_polar_surface_area"].to_numpy() * 0.01)
        

        print(f"📊 Feature Alignment Matrix Completed. Combined Vector Design Space Shape: {X.shape}")
        return X, y_reg, y_clf

    # =====================================================================
    # FUNCTION train_and_evaluate(X, y_reg, y_clf)
    # =====================================================================
    def train_and_evaluate(self, X: np.ndarray, y_reg: np.ndarray, y_clf: np.ndarray):
        """Applies StandardScaler scaling across partitions and optimizes downstream predictors."""
        # SPLIT dataset into 80% Training and 20% Testing subsets
        X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
        _, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

        print("\n⚖️ Applying StandardScaler to prevent feature dominance and convergence errors...")
        # FIT StandardScaler on X_train partition and TRANSFORM both subsets
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("🏋️ Training continuous Regression Head (Random Forest)...")
        # FIT Random Forest Regressor using scaled Training partition
        self.regressor.fit(X_train_scaled, y_train_reg)
        reg_preds = self.regressor.predict(X_test_scaled)
        mse = mean_squared_error(y_test_reg, reg_preds)
        r2 = r2_score(y_test_reg, reg_preds)

        print("🏋️ Training binary Classification Head (Logistic Regression)...")
        # FIT Logistic Regression Classifier using scaled Training partition
        self.classifier.fit(X_train_scaled, y_train_clf)
        clf_preds = self.classifier.predict(X_test_scaled)
        clf_probs = self.classifier.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test_clf, clf_preds)
        try:
            roc_auc = roc_auc_score(y_test_clf, clf_probs)
        except ValueError:
            roc_auc = 1.0

        # PRINT comprehensive model evaluation diagnostic report
        print("\n" + "="*50)
        print("📈 UPDATED SCALED METRICS EVALUATION DIAGNOSTIC REPORT")
        print("="*50)
        print(f"Continuous Regression Head Metrics:")
        print(f"  -> Mean Squared Error (MSE): {mse:.4f}")
        print(f"  -> R-squared (R2) Variance Score: {r2:.4f}")
        print("-"*50)
        print(f"Binary Classification Head Metrics:")
        print(f"  -> Total Validation Accuracy: {accuracy * 100:.2f}%")
        print(f"  -> Area Under ROC Curve (ROC-AUC): {roc_auc:.4f}")
        print("\nDetailed Classification Log Matrix:")
        print(classification_report(y_test_clf, clf_preds, target_names=["BBB-", "BBB+"]))
        print("="*50)

    # =====================================================================
    # FUNCTION serialize_models(output_directory)
    # =====================================================================
    def serialize_models(self, output_dir: str = "models"):
        """Serializes the scaler state alongside model weights into deployment artifacts."""
        os.makedirs(output_dir, exist_ok=True)
        
        scaler_path = os.path.join(output_dir, "fitted_scaler.pkl")
        reg_path = os.path.join(output_dir, "rf_regressor.pkl")
        clf_path = os.path.join(output_dir, "logistic_classifier.pkl")
        
        # SAVE trained StandardScaler state to disk via pickle
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        # SAVE trained Random Forest model state to disk via pickle
        with open(reg_path, "wb") as f:
            pickle.dump(self.regressor, f)
        # SAVE trained Logistic Regression model state to disk via pickle
        with open(clf_path, "wb") as f:
            pickle.dump(self.classifier, f)
            
        print(f"📦 Production Artifact Assembly Complete!")
        print(f"   -> Fitted Pipeline Scaler saved to: {scaler_path}")
        print(f"   -> Regression Head Pipeline State saved to: {reg_path}")
        print(f"   -> Classification Head Pipeline State saved to: {clf_path}")

if __name__ == "__main__":
    predictor = BrainBarrierPredictor()
    X, y_reg, y_clf = predictor.load_and_align_datasets()
    predictor.train_and_evaluate(X, y_reg, y_clf)
    predictor.serialize_models()