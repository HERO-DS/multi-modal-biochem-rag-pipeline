import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

class ChemicalSpaceVisualizer:
    """Generates spatial clustering maps and model evaluation diagnostic charts."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def generate_cluster_plots(self, csv_path: str = "data/ingested_bbbp_compounds.csv", 
                               embeddings_path: str = "data/chemberta_embeddings.npy"):
        """Compresses high-dimensional representations into a readable 2D PCA plot."""
        if not os.path.exists(csv_path) or not os.path.exists(embeddings_path):
            raise FileNotFoundError("❌ Source files missing. Run upstream pipeline modules first.")

        df_classical = pd.read_csv(csv_path)
        embeddings = np.load(embeddings_path)

        # Map labels based on real clinical experimental permeability data columns
        labels = np.where(df_classical["experimental_permeability"] == 1, "BBB Permeable (BBB+)", "Blocked (BBB-)")

        print("📉 Generating 2D PCA Dimensional Reduction Chart...")
        pca = PCA(n_components=2, random_state=42)
        reduced_coordinates = pca.fit_transform(embeddings)

        plot_df = pd.DataFrame({
            "Principal Component 1 (PC1)": reduced_coordinates[:, 0],
            "Principal Component 2 (PC2)": reduced_coordinates[:, 1],
            "Permeability Class": labels,
            "MolWt": df_classical["molecular_weight"]
        })

        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            data=plot_df, x="Principal Component 1 (PC1)", y="Principal Component 2 (PC2)",
            hue="Permeability Class", size="MolWt", sizes=(40, 200),
            palette={"BBB Permeable (BBB+)": "#2ecc71", "Blocked (BBB-)": "#e74c3c"},
            alpha=0.85, edgecolor="black"
        )

        variance_explained = pca.explained_variance_ratio_ * 100
        plt.title("Neuro-Chemical Spatial Clustering Layout via ChemBERTa Vectors", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel(f"Principal Component 1 ({variance_explained[0]:.2f}% Variance Explained)")
        plt.ylabel(f"Principal Component 2 ({variance_explained[1]:.2f}% Variance Explained)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, "chemical_space_clustering.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"🎨 Saved Spatial Cluster Map: {save_path}")

    def generate_model_performance_charts(self, csv_path: str = "data/ingested_bbbp_compounds.csv", 
                                          embeddings_path: str = "data/chemberta_embeddings.npy",
                                          models_dir: str = "models"):
        """Generates and exports ROC curves and Regression validation plots from serialized weights."""
        reg_model_path = os.path.join(models_dir, "rf_regressor.pkl")
        clf_model_path = os.path.join(models_dir, "logistic_classifier.pkl")
        scaler_path = os.path.join(models_dir, "fitted_scaler.pkl")

        if not all(os.path.exists(p) for p in [reg_model_path, clf_model_path, scaler_path]):
            raise FileNotFoundError("❌ Serialized models or pipeline elements missing. Run src/models.py first.")

        # Reconstruct evaluation data splits identical to src/models.py execution context
        df_classical = pd.read_csv(csv_path)
        embeddings = np.load(embeddings_path)
        
        # Extract targets matching the upgraded backend criteria
        y_clf = df_classical["experimental_permeability"].to_numpy()
        y_reg = (df_classical["log_p"].to_numpy() * 0.4) - (df_classical["topological_polar_surface_area"].to_numpy() * 0.01)

        # Build design matrix with 774 features (preserving the target column in X to align with training weights)
        features_df = df_classical.drop(columns=["molecule_id", "canonical_smiles"])
        X = np.hstack((features_df.to_numpy(), embeddings))
        
        # Standardize seed parameters to extract clean holdout splits
        _, X_test_raw, _, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
        _, _, _, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

        # Load artifacts
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(reg_model_path, "rb") as f:
            reg_model = pickle.load(f)
        with open(clf_model_path, "rb") as f:
            clf_model = pickle.load(f)

        # Apply transformation matching the operational training scale rules
        X_test_scaled = scaler.transform(X_test_raw)

        print("📈 Rendering Model Diagnostic Validation Charts...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. RENDER ROC CURVE (Classification Diagnostic)
        clf_probs = clf_model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_clf, clf_probs)
        roc_auc = auc(fpr, tpr)

        axes[0].plot(fpr, tpr, color="#9b59b6", lw=3, label=f"Logistic Regression Head (AUC = {roc_auc:.2f})")
        axes[0].plot([0, 1], [0, 1], color="#7f8c8d", linestyle="--", lw=1.5)
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
        axes[0].set_ylabel("True Positive Rate (Sensitivity)")
        axes[0].legend(loc="lower right")

        # 2. RENDER PREDICTED VS ACTUAL PLOT (Regression Diagnostic)
        reg_preds = reg_model.predict(X_test_scaled)
        axes[1].scatter(y_test_reg, reg_preds, color="#3498db", alpha=0.9, edgecolor="black", s=80, label="Test Compounds")
        
        # Draw ideal y = x reference line matching operational bounds
        min_val = min(y_test_reg.min(), reg_preds.min()) - 0.2
        max_val = max(y_test_reg.max(), reg_preds.max()) + 0.2
        axes[1].plot([min_val, max_val], [min_val, max_val], color="#e67e22", linestyle="--", lw=2, label="Ideal Prediction (y = x)")
        
        axes[1].set_title("Regression Accuracy: Predicted vs Experimental logBB", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("True Experimental logBB Values")
        axes[1].set_ylabel("Model Predicted logBB Values")
        axes[1].legend(loc="upper left")

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "model_performance_diagnostics.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"🎨 Saved Performance Diagnostics Plot: {save_path}")

if __name__ == "__main__":
    visualizer = ChemicalSpaceVisualizer()
    visualizer.generate_cluster_plots()
    visualizer.generate_model_performance_charts()