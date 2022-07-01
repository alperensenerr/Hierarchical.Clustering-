from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

veriler = pd.read_csv('musteriler.csv')

ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean',linkage='ward')
data = veriler.iloc[:,2:].values
y_tahmin = ac.fit_predict(data)

plt.scatter(data[y_tahmin==0,0],data[y_tahmin==0,1],s=100, c='red')
plt.scatter(data[y_tahmin==1,0],data[y_tahmin==1,1],s=100, c='green')
plt.scatter(data[y_tahmin==2,0],data[y_tahmin==2,1],s=100, c='blue')
plt.scatter(data[y_tahmin==3,0],data[y_tahmin==3,1],s=100, c='yellow')
plt.show()

dendrogram = sch.dendrogram(sch.linkage(data, method='ward'))
plt.show()
#we could see from dendrogram the optimum cluster number  is 4
