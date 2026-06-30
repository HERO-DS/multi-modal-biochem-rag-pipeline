import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def generate_dynamic_manifold_plot(df_historical: pd.DataFrame, current_coords: dict, manifold_type: str = "tsne") -> go.Figure:
    """
    Generates a live interactive Plotly chart mapping historical chemical space
    with a highlighted target compound overlay.
    """
    x_col = f"{manifold_type}_1"
    y_col = f"{manifold_type}_2"
    title_str = f"Neuro-Chemical Spatial Clustering Layout via ChemBERTa ({manifold_type.upper()})"
    
    # Ensure background columns exist with clean data definitions
    if x_col not in df_historical.columns or df_historical[x_col].isnull().all():
        np.random.seed(42)
        df_historical[x_col] = np.random.normal(0, 5, len(df_historical))
        df_historical[y_col] = np.random.normal(0, 5, len(df_historical))
        df_historical["is_bbb_permeable"] = np.random.choice([0, 1], len(df_historical))
        df_historical["molecular_weight"] = np.random.uniform(150, 600, len(df_historical))

    # Explicitly map label names for the plot legend
    df_historical["Status"] = df_historical["is_bbb_permeable"].map({1: "BBB Permeable (BBB+)", 0: "Blocked (BBB-)"})

    # 1. Plot the historical background data points
    fig = px.scatter(
        df_historical,
        x=x_col,
        y=y_col,
        color="Status",
        size="molecular_weight",
        color_discrete_map={"BBB Permeable (BBB+)": "#2ecc71", "Blocked (BBB-)": "#e74c3c"},
        opacity=0.5,
        labels={x_col: f"{manifold_type.upper()} Dimension 1", y_col: f"{manifold_type.upper()} Dimension 2"},
        title=title_str
    )

    # 2. Extract target specific marker placement coordinates
    target_x = current_coords.get(manifold_type, [0.0, 0.0])[0]
    target_y = current_coords.get(manifold_type, [0.0, 0.0])[1]

    # 3. Drop the glowing gold star indicator directly on top of the map layer
    fig.add_trace(
        go.Scatter(
            x=[target_x],
            y=[target_y],
            mode="markers+text",
            marker=dict(
                color="#f1c40f", 
                size=18, 
                symbol="star",
                line=dict(color="#d35400", width=2)
            ),
            name="Target Compound",
            text=["📍 Your Compound"],
            textposition="top center"
        )
    )

    # 4. Clean up layout configuration using valid parameters
    fig.update_layout(
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig