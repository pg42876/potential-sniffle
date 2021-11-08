#Imports
import pandas as pd
from si.data import Dataset
from si.util.util import summary
from si.util.scale import StandardScaler #Standard Scale
from si.data.feature_selection import f_regression, SelectKBest, VarianceThreshold #Feature Selection
from si.unsupervised.Clustering import Kmeans #Clustering
import matplotlib.pyplot as plt
from si.unsupervised.Clustering import PCA #PCA
import os

DIR = os.path.dirname(os.path.realpath('.'))
filename = os.path.join(DIR, 'datasets/breast-bin.data')

#Labeled Dataset
dataset = Dataset.from_data(filename, labeled = True)

print(dataset.X[:5, :])
print(dataset.Y[:5])

print("Has label:", dataset.hasLabel())
print("Number of features:", dataset.getNumFeatures())
print("Number of classes:", dataset.getNumClasses())
summary(dataset)

dataset.toDataframe()

#Standard Scaler
sc = StandardScaler()
ds2 = sc.fit_transform(dataset)
summary(ds2)

#Feature Selection - Variance Threshold
vt = VarianceThreshold(8)
ds2 = vt.fit_transform(dataset)
summary(ds2)

#Feature Selection - SelectKBest
skb = SelectKBest(5)
ds3 = skb.fit_transform(dataset)
summary(ds3)

#Clustering
filename = os.path.join(DIR, 'datasets/iris.data')
df = pd.read_csv(filename)
iris = Dataset.from_dataframe(df, ylabel = "class")

##Índice das fearures para o gráfico e respetivo gráfico
c1 = 0
c2 = 1
plt.scatter(iris.X.iloc[:,c1], iris.X.iloc[:,c2])
plt.xlabel(iris._xnames[c1])
plt.ylabel(iris._xnames[c2])
plt.show()

kmeans = Kmeans(3)
cent, clust = kmeans.fit_transform(iris)

plt.scatter(iris.X.iloc[:, c1], iris.X.iloc[:, c2], c = clust)
plt.scatter(cent[:, c1],cent[:, c2], s = 100, c = 'black', marker = 'x')
plt.xlabel(iris._xnames[c1])
plt.ylabel(iris._xnames[c2])
plt.show()

#PCA
pca = PCA(2, method = 'svd')

reduced = pca.fit_transform(iris)
print(pca.variance_transform())
iris_pca = Dataset(reduced[0], iris.Y, xnames = ['pc1', 'pc2'], yname = 'class')
iris_pca.toDataframe()

plt.scatter(iris_pca.X[:, 0], iris_pca.X[:, 1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()