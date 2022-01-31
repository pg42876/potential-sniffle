# Sistemas Inteligentes para Bioinformática

#Exemplos de validação de código. Deverá corrigir os imports e diretórios de acordo com a sua implementação.

from src.si.data.dataset import Dataset
from src.si.util.util import summary
import os

DIR = os.path.dirname(os.path.realpath('.'))
filename = os.path.join(DIR, 'SIB/datasets/breast-bin.data')

print('-----------------------Labeled Dataset--------------------------')
## Labeled dataset
dataset = Dataset.from_data(filename, labeled=True)

print(dataset.X[:5, :])
print(dataset.Y[:5])

print("Has label:", dataset.hasLabel())
print("Number of features:", dataset.getNumFeatures())
print("Number of classes:", dataset.getNumClasses())
print('Labeled dataset:')
print(summary(dataset))

print('dataset.toDataframe():')
print(dataset.toDataframe())

print('-----------------------Standard Scaler--------------------------')
## Standard Scaler

from src.si.util.scale import StandardScaler
sc = StandardScaler()
ds2 = sc.fit_transform(dataset)
print('ds2 Standard Scaler:')
print(summary(ds2))
print('------------------Feature Selection------------------------')

# Feature Selection
from src.si.data.Features_Selection import f_regression, SelectKBest, VarianceThreshold

## Variance Threshold
vt = VarianceThreshold(8)
ds2 = vt.fit_transform(dataset)
print('ds2 Variance Threshold:')
print(summary(ds2))


## SelectKBest
# SelectKBest for classification
skb = SelectKBest(5)
ds3 = skb.fit_transform(dataset)
print('ds3 SelectKBest:')
print(summary(ds3))

print('---------------------------Clustering------------------------------')
print('Plot')
# Clustering
from src.si.unsupervised.Clustering import Kmeans
import pandas as pd
import matplotlib.pyplot as plt

# o dataset iris nao estava inicialmente no github
filename = os.path.join(DIR, 'SIB/datasets/iris.data')
df = pd.read_csv(filename)
iris = Dataset.from_dataframe(df,ylabel="class")

# indice das features para o plot
c1 = 0
c2 = 1
# plot
plt.scatter(iris.X[:,c1], iris.X[:,c2])
plt.xlabel(iris._xnames[c1])
plt.ylabel(iris._xnames[c2])
plt.show()

kmeans = Kmeans(3)
cent, clust = kmeans.fit_transform(iris)

plt.scatter(iris.X[:,c1], iris.X[:,c2],c=clust)
plt.scatter(cent[:,c1],cent[:,c2], s = 100, c = 'black',marker='x')
plt.xlabel(iris._xnames[c1])
plt.ylabel(iris._xnames[c2])
plt.show()
# podem obter clusterings diferentes já que estes dependem da escolha dos centroids iniciais

print('-----------------------------PCA----------------------------')
# PCA
from src.si.unsupervised.PCA import PCA
pca = PCA(2, using='svd')

reduced = pca.fit_transform(iris)
print('pca.explained_variances():')
print(pca.explained_variances())

iris_pca = Dataset(reduced[0],iris.Y,xnames=['pc1','pc2'],yname='class')
print('iris_pcs.toDataframe():')
print(iris_pca.toDataframe())

plt.scatter(iris_pca.X[:,0], iris_pca.X[:,1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

