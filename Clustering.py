#!/usr/bin/env python
# coding: utf-8

# # K-means sur le prix et la localisation

# Import des fichiers utiles

# In[1]:


import pandas as pd
import dict_var
import geopandas as gpd
import pathlib
import geopandas as gpd
import fiona
from shapely.geometry import shape
import numpy as np


# In[2]:


var_mutation = dict_var.scrapCeremaTable("http://doc-datafoncier.cerema.fr/dv3f/doc/table/mutation")
var_disp_parcelle = dict_var.scrapCeremaTable("http://doc-datafoncier.cerema.fr/dv3f/doc/table/disposition_parcelle")
var_local = dict_var.scrapCeremaTable("http://doc-datafoncier.cerema.fr/dv3f/doc/table/local")
var_main = pd.concat([var_mutation,var_disp_parcelle,var_local])
var_main.head(10)


# import du fichier de travail

# In[3]:


df = pd.read_csv("dataset_travail")


# **sélection des variables utiles**

# In[4]:


df_kmeans = df[df["codtypbien"]==111][["idmutation","valeurfonc"]].copy()


#     import du fichier shapefile contenant les localisations des bien

# In[5]:


mut_loc = gpd.read_file("C:/Users/joris/Google Drive/ENSAE/2A/S1/PythonDataScientist/Projet/DVF+ downloaded/DVF+r84/r84/r84_mutation_geomlocmut.shp")


# **Jointure avec la base initiale**

# In[6]:


joined = mut_loc.rename(columns={"IDMUTATION":"idmutation"}).copy().set_index("idmutation").join(df_kmeans.set_index('idmutation'))


# In[7]:


joined = joined.dropna().copy()
joined.head(5)


# **Creation de la matrice du K-means**

# In[8]:


def getX(joined):
    A = joined["geometry"].tolist()
    B = joined["valeurfonc"].tolist()
    A = [[a.x,a.y] for a in A]
    A = [A[i]+[B[i]] for i in range(len(A))]
    return(A)


# In[9]:


X = getX(joined)


# import des librairies pour le kmeans

# In[10]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# **Détermination du nombre de cluster optimal**

# In[11]:


def kmeans(X):
    max_k = 10
    
## iterations
    distortions = [] 
    for i in range(1, max_k+1):
        if len(X) >= i:
           model = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
           model.fit(X)
           distortions.append(model.inertia_)
## best k: the lowest derivative
    k = [i*100 for i in np.diff(distortions,2)].index(min([i*100 for i 
     in np.diff(distortions,2)]))
## plot
    fig, ax = plt.subplots()
    ax.plot(range(1, len(distortions)+1), distortions)
    ax.axvline(k, ls='--', color="red", label="k = "+str(k))
    ax.set(title='The Elbow Method', xlabel='Number of clusters', 
           ylabel="Distortion")
    ax.legend()
    ax.grid(True)
    plt.show()
    return()


# In[12]:


kmeans(X)


# **Obtention des premiers clusters et tracé**

# In[13]:


Clusters = pd.Series(KMeans(n_clusters=6, init='k-means++', max_iter=300, n_init=10, random_state=0).fit_predict(X))


# In[14]:


Clusters.index=joined.index


# In[15]:


Clustered =pd.concat([joined,Clusters],axis=1)


# On observe des clusters très peu pertinents et lisibles, pour améliorer cela on essaie d'appliquer le kmeans sur la valeur foncière au mètre carré

# In[16]:


plt.figure(figsize=(20,20))
Clustered[["geometry",0]].plot(column=0)


# # Pondération par la surface

# In[17]:


df_kmeans_sur = df[df["codtypbien"]==111].copy()
df_kmeans_sur["valeurfoncsur"]= df_kmeans_sur["valeurfonc"]/df_kmeans_sur["sbatmai"]


# In[18]:


df_kmeans_sur["valeurfoncsur"]= df_kmeans_sur["valeurfonc"]/df_kmeans_sur["sbatmai"]


# In[19]:


def getX_sur(joined):
    A = joined["geometry"].tolist()
    B = joined["valeurfoncsur"].tolist()
    A = [[a.x,a.y] for a in A]
    A = [A[i]+[B[i]] for i in range(len(A))]
    return(A)


# In[20]:


df_kmeans_sur = df_kmeans_sur[["idmutation","valeurfoncsur"]].copy()
joined_sur = mut_loc.rename(columns={"IDMUTATION":"idmutation"}).copy().set_index("idmutation").join(df_kmeans_sur.set_index('idmutation'))
joined_sur = joined_sur.dropna().sort_values(by=["idmutation"]).copy()
X_sur = getX_sur(joined_sur)


# **On fait tourner le kmeans sur la nouvelle matrice de coordonnées**

# In[21]:


kmeans(X_sur)


# In[22]:


Clusters_sur = KMeans(n_clusters=7, init='k-means++', max_iter=300, n_init=10, random_state=0).fit_predict(X_sur)


# **Tracé des clusters**

# In[23]:


Cluster_series_sur = pd.Series(Clusters_sur)
Cluster_series_sur.index=joined_sur.index


# In[24]:


Clustered_sur =pd.concat([joined_sur,Cluster_series_sur],axis=1)
Clustered_sur =Clustered_sur.rename(columns={0:"Cluster"})


# **Les clusters sont intéressants mais quand on regarde la dispersion des prix on est peu satisfaits**

# Les moyennes sont plutôt bonnes, on distingue de bonnes zones de prix

# In[25]:


Clustered_sur.groupby("Cluster").mean('valeurfoncsur').sort_values(by='valeurfoncsur')


# In[26]:


corresp_cluster = Clustered_sur.groupby("Cluster").mean('valeurfoncsur').sort_values(by='valeurfoncsur')


# In[27]:


Clustered_sur["Cluster"] = Clustered_sur["Cluster"].apply(lambda x:corresp_cluster.loc[x])


# In[28]:


fig, ax = plt.subplots(1, 1)

plt.title("K-means sur la région Rhône-Alpes")
Clustered_sur[["geometry","Cluster"]].plot(column="Cluster",legend=True,ax=ax)


# Malheureusement les ecarts types sont très dipersés

# In[29]:


Clustered_sur.groupby("Cluster").std()


# # Raccord à la base commune

# In[30]:


data_clustered = Clustered_sur.join(df.set_index('idmutation'))[["geometry","valeurfoncsur","Cluster","l_codinsee"]].copy()


# In[31]:


def aggrege(s):
    string =""
    for i in s:
        if i!='{':
            string+=i
        if i=="," or i=="}":
            return(string[:-1])


# In[33]:


data_clustered['l_codinsee']=data_clustered['l_codinsee'].apply(aggrege)


# In[34]:


ra = gpd.read_file("communes-20190101.json")
ra = ra[ra["insee"].isin(list(data_clustered["l_codinsee"]))].copy()
ra = ra.rename(columns={"insee":"l_codinsee"}).set_index("l_codinsee")
carte = ra.join(data_clustered.groupby("l_codinsee")["Cluster"].mean()).copy()


# In[75]:


def closest_cluster(x):
    c = Clustered_sur["Cluster"].unique()
    a = np.argmin([abs(c[i]-x) for i in range(len(c))])
    
    return(c[a])


# In[79]:


carte["Cluster"] = carte["Cluster"].apply(closest_cluster)


# In[80]:


fig, ax = plt.subplots(1, 1)

plt.title("K-means sur la région Rhône-Alpes")
carte.plot(column="Cluster",legend=True,ax=ax)

