"""
K-means clustering: Tested between 1 to 51 using elbow method. 
No sharp elbow observed indicating weak clustering.
Silhouette average checked for k = 3, 5, 10. 
Hovered between 0.1 to 0.2 indicating weak clustering.
"""

import sys
from pathlib import Path
 # __file__ refers to current directory, first parent nvigates from file to its directory, second parent goes to directory to parent directory
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)
from main import *

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

df = pd.read_parquet(preprocessed_parquet_file_path)
df_vectors_dict = {}

for field_name in patent_fields:
  df_field_vectors = pd.read_parquet(d2v_vectors_file_path[field_name])
  df_vectors_dict[field_name] = df_field_vectors

df_combined_vectors = pd.read_parquet(d2v_combined_vectors_file_path) # combined vectors includes weighted composite vectors derived from TI, AB, ICLM, CLMS vectors
df_vectors_dict['combined'] = df_combined_vectors

for field_name, df_vectors in df_vectors_dict.items():
  vector_columns = df_vectors.columns[0:100]
  numpy_vectors = df_vectors[vector_columns].to_numpy()
  
  k = 5
  kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
  cluster_labels = kmeans.fit_predict(numpy_vectors)

  df_vectors[f'Clusters_{field_name}'] = cluster_labels
  df_vectors.to_excel(project_folder + excel_out + f'\kmeans\kmeans-clustered-vectors-{field_name}.xlsx')
  df_vectors.to_parquet(project_folder + parquet_out + f'\kmeans\kmeans-clustered-vectors-{field_name}.parquet')