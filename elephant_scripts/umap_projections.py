"""
This script projects the 129D acoustic embeddings into 2D latent space using the Uniform Manifold Approximation and Projection (UMAP) algorithm.

"""
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler


def z_score_transform(df):
    """Standardise each column in the dataset.

    Standardised is mean=0 variance=1.
    """
    scaler = StandardScaler()
    scaler.fit(df)
    transformed_df = pd.DataFrame(
        scaler.transform(df), columns=df.columns, index=df.index
    )
    return transformed_df


def umap_projections(transformed_df, metadata_df, N_COMP, metric, min_dist):
    """
    FIX: This function is currently doing too much.

    Args:
        results_df (pd.DataFrame): DataFrame containing the acoustic embeddings.
            TODO: Add a description of what columns are expected in this
            dataframe.
        metadata_df (pd.DataFrame): DataFrame containing metadata information.
            TODO: Similarly for this dataframe.
        N_COMP (int): Number of UMAP components.
        metric (str): Distance metric for UMAP.
        min_dist (float): Minimum distance for UMAP.

    Returns:
        pd.DataFrame: DataFrame with UMAP projections and metadata.
    """

    # BUG: The z_score_transform does not modify the dataframe in place.
    # Hence the transformed_df is not actually transformed.
    normalised = z_score_transform(transformed_df)
    # Specify UMAP set-up and parameters
    reducer = umap.UMAP(n_components=N_COMP, metric=metric, min_dist=min_dist)

    print("Acoustic features normalised")

    # Fit UMAP. Embedding contains the new coordinates of datapoints
    # WARNING: Maybe the column "recording_id" is being unadvertedly passed
    # into the UMAP algorithm. Which would explain why contigous recordings
    # appear to be close to each other in the UMAP plot.
    embedding = reducer.fit_transform(normalised.drop(["recording_id"], axis=1))

    print("Dimensionality reduction completed")

    # Add UMAP coordinates to dataframe
    # NOTE: It is better to create a new dataframe with the UMAP coordinates
    # rather than adding them to the existing dataframe.
    # results = pd.DataFrame(
    #     {
    #         "recording_id": transformed_df["recording_id"],
    #         **{f"UMAP_{i + 1}": embedding[:, i] for i in range(N_COMP)},
    #     }
    # )
    # Also, only the recording_id column is needed to link the UMAP coordinates
    # to the data of each vocalisation. Hence you avoid unnecessary duplication
    # of data (it is always better to have a single source of truth).

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

    # NOTE: The join can be done outside of this function, as it is no longer
    # part of the dimensionality reduction process.

    # Merge metadata df and results indices
    umap_df = umap_df.set_index("recording_id").join(
        metadata_df.set_index("recording_id")
    )

    # NOTE: FYI you can also use the following command to join the dataframes:
    # merged = pd.merge(
    #     left=umap_df,
    #     right=metadata_df,
    #     left_on="recording_id",
    #     right_on="recording_id",
    # )
    # Avoids having to set the index of the dataframes.

    # Store the embeddings in the umap dataframe
    return umap_df  # Return the processed results


print("Functions for Dimensionality Reduction successfully loaded")
