#!/usr/bin/env python
# coding: utf-8

# # Raccord à la base

# In[1]:


import pandas as pd
import dict_var
import geopandas as gpd
import pathlib
import geopandas as gpd
import fiona
from shapely.geometry import shape
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


ra = gpd.read_file("communes-20190101.json")
mut_loc = gpd.read_file("C:/Users/joris/Google Drive/ENSAE/2A/S1/PythonDataScientist/Projet/DVF+ downloaded/DVF+r84/r84/r84_mutation_geomlocmut.shp")
df = pd.read_csv("dataset_travail")


# In[3]:



df2 = df[df["codtypbien"]==111].copy()
df2["valeurfoncsur"]= df2["valeurfonc"]/df2["sbatmai"]
df2 = df2.set_index("idmutation")[["l_codinsee","valeurfoncsur"]]
df2 = mut_loc.rename(columns={"IDMUTATION":"idmutation"}).copy().set_index("idmutation").join(df2)
df2 = df2.dropna()


# In[4]:


qmax = df2["valeurfoncsur"].quantile(q=0.99)
qmin = df2["valeurfoncsur"].quantile(q=0.01)


# In[5]:


df3 = df2[df2["valeurfoncsur"].between(qmin,qmax)]
fig, ax = plt.subplots(1, 1)

plt.title("Valeurfoncsur région Rhône-Alpes")
df3[["geometry","valeurfoncsur"]].plot(column="valeurfoncsur",legend=True,ax=ax)


# # Regression spatiale

# In[4]:


def aggrege(s):
    string =""
    for i in s:
        if i!='{':
            string+=i
        if i=="," or i=="}":
            return(string[:-1])
df2['l_codinsee']=df2['l_codinsee'].apply(aggrege)


# In[5]:


df2["l_codinsee"]= pd.to_numeric(df2["l_codinsee"])


# In[6]:


def dummy(x,code):
    if x==code:
        return(1)
    else:
        return(0)


# In[7]:


def preprocess(df2):
    df = df2.copy()
    for code in list(df2["l_codinsee"].unique()):
        df[code] = df["l_codinsee"].apply(lambda x: dummy(x,code))
    return(df.iloc[:,3:].to_numpy(),df['valeurfoncsur'].to_numpy())


# In[8]:


X,y = preprocess(df2)


# In[10]:


import numpy as np
from sklearn.linear_model import LinearRegression


# In[11]:


reg = LinearRegression().fit(X, y)


# In[19]:


reg.coef_

