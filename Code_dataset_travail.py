#!/usr/bin/env python
# coding: utf-8

# In[2]:


import geopandas as gpd


# In[3]:


import pandas as pd
import os
import dict_var


# In[4]:


os.getcwd()


# In[5]:


pd.reset_option


# In[6]:


df = pd.read_csv(r"C:\Users\joris\Google Drive\ENSAE\2A\S1\PythonDataScientist\Projet\DVF+ downloaded\DVF+r84\r84\r84_mutation.csv")


# In[7]:


for elt in df.columns:
    print(elt)


# In[8]:


var_mutation = dict_var.scrapCeremaTable("http://doc-datafoncier.cerema.fr/dv3f/doc/table/mutation")
var_disp_parcelle = dict_var.scrapCeremaTable("http://doc-datafoncier.cerema.fr/dv3f/doc/table/disposition_parcelle")
var_local = dict_var.scrapCeremaTable("http://doc-datafoncier.cerema.fr/dv3f/doc/table/local")


# In[9]:


var_main = pd.concat([var_mutation,var_disp_parcelle,var_local])


# In[10]:


var_main.columns


# In[11]:


var_main[var_main[0].isin(df.columns)].reset_index()


# In[22]:


cleaning_list = list(var_main[var_main[0].isin(df.columns)].reset_index().iloc[[4,5,11,12,61,62,63,64,65,66,68,69,70,71],1])
cleaning_list.remove('idmutation')
cleaning_list.remove('idmutation')
cleaning_list


# In[24]:


columns = list(df.columns)


# In[25]:


def private(l1,l2):
    liste = l1
    for e in l2:
        while e in liste:
            liste.remove(e)
    return(liste)
            


# In[26]:


clean = private(columns,cleaning_list)


# In[27]:


df[clean].to_csv("dataset_travail")

