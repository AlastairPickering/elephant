"""
This script contains import statements for the libraries to be used in the classification of forest elephant vocalisations.

"""
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import matplotlib.pyplot as plt
import librosa
import umap.umap_ as umap
import sys
sys.path.insert(0, '..')
import seaborn as sns
import scipy
from scipy.spatial.distance import pdist, squareform
from IPython.display import Image, Audio, display
import random
from sklearn.neighbors import NearestNeighbors
import plotly
from plotly.offline import iplot, plot
from plotly import graph_objs as go
from sklearn import metrics
import tensorflow_hub as hub
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from collections import namedtuple
import plotly.express as px

print("Main libraries successfully imported")

