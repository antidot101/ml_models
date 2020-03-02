#!/usr/bin/env python
# coding: utf-8

# ## Метод К-средних

# In[1]:


from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np


X, y = make_blobs(n_samples=350, 
                  n_features=2, 
                  centers=5,
                  cluster_std=0.8, 
                  shuffle=True, 
                  random_state=3)
plt.scatter(X[:,0], X[:,1], c='b', marker='o', s=25)
plt.grid()


# In[2]:


from sklearn.cluster import KMeans


km = KMeans(n_clusters=5,
            init='random',
            n_init=10,
            max_iter=300,
            random_state=0)
y_km = km.fit_predict(X)
plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=km.labels_)
plt.scatter(km.cluster_centers_[:,0], 
            km.cluster_centers_[:,1], 
            marker='^', 
            c='r', 
            s=150, 
            label='Центроиды')
plt.grid()
plt.legend(fontsize=15)
plt.title('Результаты кластерного анализа', fontdict={'fontsize': 18})
print('Число итераций random_init: ', km.n_iter_)
print('Inertia: %.2f' % km.inertia_)


# In[3]:


km = KMeans(n_clusters=5,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=0)
km.fit_predict(X)
print('Число итераций k-means++: ', km.n_iter_)
print('Inertia: %.2f' % km.inertia_)


# Метод локтя для подбора числа кластеров k

# In[4]:


inert = []
for i in range(1, 11) :
    km = KMeans(n_clusters=i, 
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X)
    inert.append(km.inertia_)
plt.plot(range(1, 11), inert, marker='o')
plt.scatter(5, inert[5], marker='o', c='r', s=150, alpha=1)
plt.xticks(range(1, 11))
plt.xlabel('n_clusters', fontsize=14)
plt.ylabel('Inertia', fontsize=14)
plt.show()


# Силуэтный анализ

# In[5]:


from matplotlib import cm
from sklearn.metrics import silhouette_samples


cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
plt.figure(figsize=(8,8))
for i, с in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == с]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    plt.barh(range(y_ax_lower, y_ax_upper),
             c_silhouette_vals,
             height=1.0)
    yticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += len(c_silhouette_vals)

plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Кластер', fontsize=16)
plt.xlabel('Силуэтный коэффициент', fontsize=16)
plt.show()


# ## Иерархическая кластеризация (агломеративная, с методом полной связи)

# In[6]:


from sklearn.cluster import AgglomerativeClustering
import pandas as pd


np.random.seed(123)
X = np.random.random_sample([10, 4])*10
labels = ['obj_' + str(i) for i in range(X.shape[0])]
features = ['x_' + str(i) for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=features, index=labels)
df


# In[7]:


ac = AgglomerativeClustering(distance_threshold=0, 
                             n_clusters=None, 
                             affinity='euclidean', 
                             linkage='complete')
model_ac = ac.fit(X)
# формируем матрицу связей linkage_matrix 
counts = np.zeros(model_ac.children_.shape[0])
n_samples = len(model_ac.labels_)
for i, merge in enumerate(model_ac.children_):
    current_count = 0
    for child_idx in merge:
        if child_idx < n_samples:
            current_count += 1
        else:
            current_count += counts[child_idx - n_samples]
    counts[i] = current_count
linkage_matrix = np.column_stack([
    model_ac.children_, model_ac.distances_, counts]).astype(float)
linkage_matrix


# In[8]:


from scipy.cluster.hierarchy import dendrogram, fcluster


plt.figure(figsize=(10,5))
dendrogram(linkage_matrix, labels=labels, orientation='left', color_threshold=0)
plt.axvline(4, color="red", linestyle="--")
plt.grid()
plt.xlabel('Евклидово расстояние')
plt.show()


# In[9]:


label = fcluster(linkage_matrix, 4, criterion='distance')
print(np.unique(label))
df.loc[:,'label'] = label
df


# In[10]:


for i, group in df.groupby('label'):
    print('-' * 55)
    print('cluster {}'.format(i))
    print(group)


# ## DBSCAN (плотностная кластеризация)

# In[12]:


from sklearn.datasets import make_moons


X, y = make_moons(n_samples=200,noise=0.05, random_state=0)
plt.scatter(X[:,0], X[:,1])
plt.show()


# In[13]:


from sklearn.cluster import DBSCAN


db = DBSCAN(eps=0.2,
            min_samples=5,
            metric='euclidean')
y_db = db.fit_predict(X)
plt.scatter(X[y_db==0,0], X[y_db==0,1],
            c='lightblue', marker='o',
            s=40, label='кластер 1')
plt.scatter(X[y_db==1,0], X[y_db==1,1],
            c='red', marker='s',
            s=40, label='кластер 2')
plt.legend()
plt.show()

