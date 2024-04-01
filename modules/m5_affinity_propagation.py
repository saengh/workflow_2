from m1_main import *

import pandas as pd
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

df = pd.read_pickle(bert_embeddings_path)
bert_embeddings = df['bert_embeddings'].tolist()

# Scale the data
scaler = StandardScaler()
scaled_embeddings = scaler.fit_transform(bert_embeddings)

# Dimensionality reduction with UMAP
umap_embeddings = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine').fit_transform(scaled_embeddings)

# Perform Affinity Propagation clustering
aff_prop = AffinityPropagation(random_state=2)
cluster_labels = aff_prop.fit_predict(umap_embeddings)
df['cluster_aff_prop'] = cluster_labels

df.to_excel(workflow_folder + r'\excel\ap_clusters.xlsx', index=False)
df.to_pickle(workflow_folder + r'\pickle\ap_clusters.pickle')

# Visualize clusters
plt.figure(figsize=(10, 8))
scatter_umap = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.title('Affinity Propagation Clustering with UMAP')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend(*scatter_umap.legend_elements(), title='Clusters', loc='upper right')
plt.savefig(workflow_folder + r'\visuals\ap_clustering.png')