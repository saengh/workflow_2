"""
Weak overlapping clusters found. 
Low silhouette scores. 
Elbow found at 11. 
k tested between 2 to 76.
"""

from m1_main import *

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
from scipy import stats

def cluster(embeddings, k):
  
  kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=5)
  return kmeans, kmeans.fit_predict(embeddings)

def find_clusters(embeddings, lower_limit, upper_limit):

  with open(workflow_folder + r'\visuals\kmeans_silhouette_scores.txt', 'w') as file: # For silhouette score outputs  
    # Choosing the number of clusters (k) - Elbow Method (simplified)
    wcss = []
    test_k_range = range(lower_limit, upper_limit)
    for k in test_k_range:
      kmeans, cluster_labels = cluster(embeddings, k)
      wcss.append(kmeans.inertia_)
      
  # Evaluating the clusters
      silhouette_avg = silhouette_score(embeddings, cluster_labels)
      print(f'Silhouette Score for k = {k}: {silhouette_avg}', file=file)

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
  plt.savefig(workflow_folder + r'\visuals\kmeans_elbow.png')

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

df = pd.read_pickle(bert_embeddings_path)
bert_embeddings = df['bert_embeddings'].tolist()

find_clusters(bert_embeddings, 2, 76)
kmeans, df['kmeans_cluster'] = cluster(bert_embeddings, 11)

df.to_excel(workflow_folder + r'\excel\kmeans_clusters.xlsx', index=False)
df.to_pickle(workflow_folder + r'\pickle\kmeans_clusters.pickle')