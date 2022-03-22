# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 21:37:45 2021

@author: Pablo
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import os

dir_principal = os.getcwd()
dir_datos = dir_principal+'\\datos'

covid = gpd.read_file(dir_datos+'/covid_periodos.shp', index = True)
covid = covid.set_index(['link','mes']).sort_index(level = 0)
covid = covid.loc[pd.IndexSlice[:,'2020-03':],:]
covid = covid.to_crs('POSGAR94')

# Separamos los campos geometricos del dataframe
geo = covid.loc[pd.IndexSlice[:,'2021-01'],'geometry']
geo = geo.reset_index(level = 'mes', drop = True)
centroides = covid.loc[pd.IndexSlice[:,'2021-01'],'geometry'].to_crs('POSGAR94').centroid
centroides = centroides.reset_index(level = 'mes', drop = True)
print("las cordenadas CRS son: "+str(geo.crs))
codiprov = covid.loc[pd.IndexSlice[:,'2021-01'],['codpcia','departamen','provincia']]


columnas = ['clasificac', 'fallecido']

# Variables acumuladas a partir del mes que todas tienen al menos 1 

covid_acum = covid[columnas].groupby(covid.index.get_level_values(0)).cumsum()
# buscamos el mes en que todos los dptos tienen al menos 1 contagio
mes = 0
valor = True
while valor == True:
    Mes = covid.index.get_level_values(1).unique()[mes]
    valor = np.any(covid_acum.loc[pd.IndexSlice[:,Mes],'clasificac'] == 0)
    mes +=1
print("El mes desde el cuál todos los dptos tienen al menos 1 contagiado es: "+str(Mes))
covid_acum['personas'] = covid.personas

covid2 = covid_acum.loc[pd.IndexSlice[:,Mes:],:]
covid_ult_mes = covid_acum.loc[pd.IndexSlice[:,'2021-07'],:]
covid_ult_mes = covid_ult_mes.reset_index(level = 'mes', drop = True)

#casos cada 10 mil habitantes
fallecidos = covid2.fallecido/(covid.loc[pd.IndexSlice[:,Mes:],:].personas/10000)
positivos = covid2.clasificac/(covid.loc[pd.IndexSlice[:,Mes:],:].personas/10000)
falle = covid2.fallecido/(covid2.personas/10000)

# Calculamos el coeficiente de localización
from lq import *
lq_ = lq(covid2,'fallecido','clasificac')
lq_fall_conf = lq_[2]
ind_fall_conf = lq_[0]

# #la variable se elige para comparar con diferentes opciones
# variable = fallecidos #covid2[['clasificac','personas']]

#%%

import elegir_modelo as em 
import procesos 
ae = em.clustering_autoencoder(centroides, n_encoders = 5)

import sklearn.cluster as  clust

km = clust.KMeans(20)
datos = covid_acum.apply(lambda x: x/covid_acum.personas*10000)[['clasificac','fallecido']]
peri = procesos.peri_columna()
x = peri.fit_transform(datos)
km.fit(x)
variab = ['clasificac','fallecido']

km_total = evaluaciones_lq(km.labels_,variab, 'personas')
km_total.calcular_indices(covid_acum)

rdo = ae.fit_autoencoder(datos[variab], n_variables= 2)
km.fit(rdo)

km_ae = evaluaciones_lq(km.labels_,variab, 'personas')
km_ae.calcular_indices(covid_acum)

covid_acum_geo = covid_ult_mes.copy()
covid_acum_geo['geometry'] = geo
gpd.GeoDataFrame(covid_acum_geo).plot(km.labels_, figsize = (10,8), legend = True)
#%%

from  sklearn.preprocessing  import PolynomialFeatures

poly = PolynomialFeatures(2)
poly.fit_transform(covid_acum[variab]).shape
covid_acum.shape
#%%

encoded = pd.DataFrame(ae.encoded_valores, columns = ['col1','col2','col3','col4','col5'])
encoded['geometry'] = geo.values
encoded = gpd.GeoDataFrame(encoded)


encoded.to_file("Geodabd/encoded.shp")
#%%

ae_prov = em.clustering_autoencoder(centroides,n_encoders = 8)
rdo2 = ae_prov.fit_autoencoder_grupo(covid_acum,codiprov[['codpcia']].values)
km = clust.KMeans(30)

km.fit(rdo2)

km_ae = evaluaciones_lq(km.labels_,variab, 'personas')
km_ae.calcular_indices(covid_acum)

gpd.GeoDataFrame(covid_acum_geo).plot(km.labels_, figsize = (10,8), legend = True)
