# Para la visualización
import matplotlib.pyplot as plt
# De apoyo
import seaborn as sns; sns.set()
# Numpy
import numpy as np
# sklearn
from sklearn.cluster import KMeans
# Matriz de confusión
from sklearn.metrics import confusion_matrix
# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Generador
from sklearn.datasets.samples_generator import make_blobs

X, Y = make_blobs(n_samples=300, centers=4)

plt.scatter(X[:,0], X[:,1], s=50)

kmeans = KMeans(n_clusters=4)

# Lo entrenamos
# Como es NO supervisado no necesita las etiquetas
kmeans.fit(X)

# Clusters
y_means = kmeans.predict(X)

# Centroides
centers = kmeans.cluster_centers_
print(centers)

# Visualizar datos
plt.scatter(X[:,0], X[:,1], c=y_means, cmap='viridis')

# Incluimos los centroides
plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.5)

plt.show()
