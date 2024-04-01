from m1_main import *

import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

df = pd.read_pickle(bert_embeddings_path)
bert_embeddings = df['bert_embeddings'].tolist()

# Scale the embeddings
scaler = MinMaxScaler()
scaled_embeddings = scaler.fit_transform(bert_embeddings)

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.9, min_samples=10)
df['dbscan_cluster'] = dbscan.fit_predict(scaled_embeddings)

df.to_excel(workflow_folder + r'\excel\dbscan_clusters.xlsx', index=False)
df.to_pickle(workflow_folder + r'\pickle\dbscan_clusters.pickle')

# Perform PCA for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_embeddings)

# Create a DataFrame for visualization
df_pca = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])

plt.figure(figsize=(10, 8))
scatter = plt.scatter(df_pca['PCA1'], df_pca['PCA2'], c=df['dbscan_cluster'], cmap='viridis', alpha=0.6)
plt.title('DBSCAN Clustering')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(*scatter.legend_elements(), title='Clusters', loc='upper right')
plt.savefig(workflow_folder + r'\visuals\dbscan_clustering.png')