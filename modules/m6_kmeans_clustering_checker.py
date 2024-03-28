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
from scipy import stats

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

  with open(project_folder + visuals_out + f'\kmeans\silhouette-scores-{field_name}.txt', 'w') as file: # For silhouette score outputs

  # Choosing the number of clusters (k) - Elbow Method (simplified)
    wcss = []
    test_k_range = range(5, 76)
    for i in test_k_range:  # Trying k from 5 to 75
      kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
      cluster_labels = kmeans.fit_predict(numpy_vectors)
      kmeans.fit(numpy_vectors)
      wcss.append(kmeans.inertia_)

  # Evaluating the clusters
      silhouette_avg = silhouette_score(numpy_vectors, cluster_labels)
      print(f'Silhouette Score for k = {i}: {silhouette_avg}', file=file)

  # Plot the results to find the "elbow"
  plt.figure(figsize=(10,6), dpi=1200)
  plt.plot(test_k_range, wcss)
  plt.grid()
  plt.title('Elbow Method')
  plt.xlabel('Number of clusters')
  plt.ylabel('WCSS')
  plt.xticks(test_k_range, rotation=90)
  
  slope, intercept, r_value, p_value, std_err = stats.linregress(test_k_range, wcss)
  plt.plot(test_k_range, slope*test_k_range + intercept, 'r', label='Fitted line')
    
  plt.tight_layout()
  plt.savefig(project_folder + visuals_out + f'\kmeans\kmeans_elbow_{field_name}.png')

