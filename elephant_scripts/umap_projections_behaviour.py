"""
This script projects the 129D acoustic embeddings into 1D latent space using the Uniform Manifold Approximation and Projection (UMAP) algorithm.

"""
import umap
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def z_score_transform(df):
    # Standardise each column in the dataset. Standardised is mean=0 variance=1.
    scaler = StandardScaler()
    scaler.fit(df)
    transformed_df = pd.DataFrame(
        scaler.transform(df), columns=df.columns, index=df.index
    )
    return transformed_df


def umap_projections(transformed_df, metadata_df, N_COMP, metric, min_dist):
    """
    Args:
        results_df (pd.DataFrame): DataFrame containing the acoustic embeddings.
        metadata_df (pd.DataFrame): DataFrame containing metadata information.
        N_COMP (int): Number of UMAP components.
        metric (str): Distance metric for UMAP.
        min_dist (float): Minimum distance for UMAP.

    Returns:
        pd.DataFrame: DataFrame with UMAP projections and metadata.
    """

    z_score_transform(transformed_df)
    # Specify UMAP set-up and parameters
    reducer = umap.UMAP(n_components=N_COMP, metric=metric, min_dist=min_dist)

    print("Acoustic features normalised")

    # Fit UMAP. Embedding contains the new coordinates of datapoints in 1D space
    embedding = reducer.fit_transform(transformed_df)

    print("Dimensionality reduction completed")

    # Add UMAP coordinates to dataframe
    for i in range(N_COMP):
        transformed_df["UMAP" + str(i + 1)] = embedding[:, i]

    # drop VGGish feature columns now that UMAP has been run on them
    results = transformed_df[
        transformed_df.columns.drop(
            list(transformed_df.filter(regex="feature_"))
        )
    ]
    results = results.drop(["duration"], axis=1)
    umap_df = results

    # Merge metadata df and results indices
    umap_df = umap_df.join(metadata_df.set_index("filename"))

    # Store the embeddings in the umap dataframe
    return umap_df  # Return the processed results


print("Functions for Dimensionality Reduction successfully loaded")
