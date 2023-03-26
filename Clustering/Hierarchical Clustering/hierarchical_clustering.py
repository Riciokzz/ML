# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:, [3, 4]].values

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method="ward"))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Training the Hierarchical Clustering model on the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="ward")
y_hc = hc.fit_predict(x)
print(y_hc)

# Visualising the clusters
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s=100, color="red", label="Cluster 1")
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s=100, color="blue", label="Cluster 2")
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s=100, color="green", label="Cluster 3")
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s=100, color="cyan", label="Cluster 4")
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s=100, color="magenta", label="Cluster 5")
plt.title("Cluster of Customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
