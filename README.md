# Decoding Elephant Vocalisations Using Unsupervised Learning
MSc project exploring elephant vocalisations using unsupervised learning

This repository contains the code needed to reproduce all of the analysis in the dissertation "Decoding Elephant Vocalisations Using Unsupervised Learning". The code is split into two notebooks, one for each of the two questions examined in the dissertation.

Question 1: Can the unsupervised methods accurately classify the rumble, roar, and trumpet call-types?

Question 2: Can the unsupervised methods identify behavioural context from rumble vocalisations?

The repository is structured as follows:

**elephant/** This top level folders contains the two Jupyter notebooks needed to run the code for the two questions. 

**elephant/audio_dir** A sample of 9 audio files from the analysis. These were recorded by the [Elephant Listening Project](https://www.elephantlisteningproject.org/)

**elephant/data** A sample of the metadata for the 9 files. The metadata is provided by the [Elephant Listening Project](https://www.elephantlisteningproject.org/). 

**elephant/elephant_scripts** The scripts needed to run the analysis in the notebook. Scripts with a '_behaviour' suffix are modified versions used to run the Q2 notebook. Scripts include:

  import_libraries.py and import_libraries_behaviour.py - main python libraries needed for analysis
  
  load_data.py and load_data_behaviour.py - scripts to import data and link audio files to metadata. This is based on the code provided by [Thomas et al 2022](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/1365-2656.13754)
  
  feature_extraction.py and feature_extraction_behaviour.py - functions to preprocess audio data and extract vggish features from it. This is heavily based on the official [VGGish documentation](https://github.com/tensorflow/models/tree/master/research/audioset/vggish)

  umap_projections.py and umap_projections_behaviour.py - functions to z-score normalise acoustic embeddings and project into low dimension space using UMAP

  silhouette_score - functions to calculate call-type silhouette scores (Q1)
  
  behaviour_stats - statistical model used to conduct behavioural context analysis (Q2)

  
  
