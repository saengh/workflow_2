from m1_main import *

import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

df = pd.read_pickle(bert_embeddings_path)
bert_embeddings = df['bert_embeddings'].tolist()

# Hierarchical Clustering
n_clusters = 7
clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
cluster_labels = clustering.fit_predict(bert_embeddings)
df['hc_cluster'] = cluster_labels

df.to_excel(workflow_folder + r'\excel\hc_clusters.xlsx', index=False)
df.to_pickle(workflow_folder + r'\pickle\hc_clusters.pickle')

# Visualization
pca = PCA(n_components=2)
doc_topic_2d = pca.fit_transform(bert_embeddings)

plt.figure(figsize=(10, 8))
for cluster_num in range(n_clusters):
    plt.scatter(doc_topic_2d[cluster_labels == cluster_num, 0],
                doc_topic_2d[cluster_labels == cluster_num, 1],
                label=f'Cluster {cluster_num + 1}')

plt.title('Hierarchical Clustering of Patents')
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.legend()
plt.savefig(workflow_folder + r'\visuals\hc_clusters.png')