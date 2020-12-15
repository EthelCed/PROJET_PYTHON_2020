#!/usr/bin/env python
# coding: utf-8

# In[1]:


import urllib
import bs4 
import pandas as pd


# In[2]:


def scrapCeremaTable(url):
    url_cerema = url
    from urllib import request
    request_text = request.urlopen(url_cerema).read()
    page = bs4.BeautifulSoup(request_text, "lxml")
    table_cerema = page.find('table')
    rows = table_cerema.find_all('tr')
    dico_cerema = dict()
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        if len(cols) > 0 : 
            dico_cerema[cols[0]] = cols[1:]
    df = pd.DataFrame.from_dict(dico_cerema,orient='index')
    return(df)


# In[3]:


df_mutation = scrapCeremaTable("http://doc-datafoncier.cerema.fr/dv3f/doc/table/mutation")
df_disp_parcelle = scrapCeremaTable("http://doc-datafoncier.cerema.fr/dv3f/doc/table/disposition_parcelle")
df_local = scrapCeremaTable("http://doc-datafoncier.cerema.fr/dv3f/doc/table/local")

