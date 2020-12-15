"""
Ceci n'est pas un code à lire. Ce sont mes essais de webscrapping sur le site seloger.com
J'ai rencontré plusieurs problèmes.
Le webscrapping sur site internet standart est assez simple avec BeautifulSoup, par exemple le site wikipédia (cf mon projet de l'année dernière)
Ici, la page web étant dynamique, j'ai dû apprendre à utiliser Selenium.
Après mettre formé sur les fonctions de base de ce module, j'ai rencontré un problème : le site seloger.com me bloquer l'accès (error 401)
S'en est suivi 4 longues heures pour tenter de contourner la protection du site : création de cookies artificiel, changement d'adresse IP, et pleins d'astuces sur internet, visiblement obsolete.
J'ai fini par renoncé et j'ai téléchargé deux bases de données en opensource sur internet 
Les deux fichiers sont "Housing_Project1_HousingSimpleRegression.py" et "Housing_Project.py"
"""


import pandas as pd
import requests
from urllib.request import Request, urlopen

url = 'https://en.wikipedia.org/wiki/S%26P_500_Index'

req = Request(url , headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}
)

webpage = urlopen(req).read()

import os
import os.path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import requests
from bs4 import BeautifulSoup

PATH = "C:\Program Files (x86)\chromedriver.exe"
URL = "https://www.seloger.com/list.htm?projects=2,5&types=1,2&natures=1,2,4&places=[{%22inseeCodes%22:[380397]}]&enterprise=0&qsVersion=1.0&m=search_hp_last"

driver = webdriver.Chrome(PATH)

driver.get(URL)
print(driver.title)

##proxy request

# Load webdriver
from selenium import webdriver

# Load proxy option
from selenium.webdriver.common.proxy import Proxy, ProxyType

# Configure Proxy Option
prox = Proxy()
prox.proxy_type = ProxyType.MANUAL

# Proxy IP & Port
prox.http_proxy = “0.0.0.0:00000”
prox.socks_proxy = “0.0.0.0:00000”
prox.ssl_proxy = “0.0.0.0:00000”

# Configure capabilities
capabilities = webdriver.DesiredCapabilities.CHROME
prox.add_to_capabilities(capabilities)

# Configure ChromeOptions
driver = webdriver.Chrome(executable_path='/usr/local/share chromedriver',desired_capabilities=capabilities)

# Verify proxy ip
driver.get("http://www.whatsmyip.org/")

##

from http_request_randomizer.requests.proxy.requestProxy import RequestProxy

req_proxy = RequestProxy()

proxies = req_proxy.get_proxy_list()

prox_fr = []

for proxy in proxies:
    if proxy.country == "France":
        prox_fr.append(proxy)


PROXY = prox_fr[4].get_address()

webdriver.DesiredCapabilities.CHROME["proxy"]={
    "httpProxy":PROXY,
    "ftpProxy":PROXY,
    "sslProxy":PROXY,

    "proxyType":"MANUAL",

}

PATH = "C:\Program Files (x86)\chromedriver.exe"
driver = webdriver.Chrome(executable_path=PATH, )

driver.get("https://www.seloger.com/list.htm?tri=initial&enterprise=0&idtypebien=2,1&idtt=2,5&naturebien=1,2,4&ci=380397&m=search_hp_new")


#time.sleep(15)
#driver.close()


##

import pandas as pd

import os
import os.path

from selenium import webdriver

import time

PATH = "C:\Program Files (x86)\chromedriver.exe"
driver = webdriver.Chrome(PATH)

driver.get("https://www.google.com/")
time.sleep(4)

link = driver.find_element_by_id("introAgreeButton")
print(link)

#time.sleep(5)
#driver.close()
