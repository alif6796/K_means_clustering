from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt

iris=datasets.load_iris()
X=iris.data

plt.scatter(X[:,0],X[:,1])
plt.show()

kmeans=KMeans(n_clusters=3)
kmeans.fit(X)
y_kmeans=kmeans.predict(X)
plt.scatter(X[:,0],X[:,1],c=y_kmeans, cmap='viridis')
plt.show()
