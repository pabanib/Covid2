# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 20:19:03 2021

@author: Pablo
"""
import numpy as np 
import pandas as pd 
import geopandas as gpd
from sklearn.pipeline import Pipeline

archivo = 'E:/Archivos/Codigos/Regionalización/datos/mendo_clust.shp'

datos = gpd.read_file(archivo)

columns = ['Bajo_st','Alto_st','Alt_st2','Jov_bnd','PrsMy25','PrsMy18']

import elegir_modelo as em 
import procesos

centroides = datos.to_crs('POSGAR94').centroid

ae_mendo = em.clustering_autoencoder(centroides, n_encoders= 2)

ae_mendo.pipe = Pipeline([('coord', procesos.agrega_centroides(centroides)),
                          ('norm', ae_mendo.norm_l1)
                          ]
    )

rdo = ae_mendo.fit_autoencoder(datos[columns], 1)

from sklearn.cluster import KMeans

km = KMeans(35,init = 'k-means++')

km.fit(rdo)

datos['ae_km'] = km.labels_

import lq

pares = {1:['Bajo_st','PrsMy25'],2:['Alto_st','PrsMy25'],3:['Alt_st2','PrsMy18'],4:['Jov_bnd','PrsMy18']}
grupos = ['Hierarch','Spect', 'SCHC', 'Skater', 'redcap', 'azp', 'maxp','km','kmed','ae_km']

lqGlobal = lq.lq_multiple(datos[columns],pares) 
rdo_lq = {}
for i in grupos:
    r = lqGlobal.calcular_indice_debil(datos[i])
    rdo_lq[i] = r
pd.DataFrame(list(rdo_lq.values()),index = list(rdo_lq.keys()))

lq.SSD(datos, 'ae_km', columns)

from sklearn import metrics as met

df_norm = ae_mendo.norm_l1.fit_transform(datos[columns])

met.silhouette_score(df_norm, km.labels_)

met.silhouette_score(rdo, km.labels_)

met.silhouette_score(df_norm, datos['azp'])
met.silhouette_score(datos[columns], datos['azp'])

geoda_rdo = pd.DataFrame(ae_mendo.encoded_valores, columns = ['var1','var2'])

geoda_rdo['geometry'] = datos.geometry

geoda_rdo = gpd.GeoDataFrame(geoda_rdo)

geoda_rdo.to_file('Geodabd/ae_mend.shp')

archivo = 'df_r/autoencod.shp'

ae_azp = gpd.read_file(archivo)
ae_azp.columns
lqGlobal.calcular_indice_debil(ae_azp['CL'])
lqGlobal.calcular_indice_debil(ae_azp['k_med'])
lqGlobal.calcular_indice_debil(ae_azp['skatter'])
lqGlobal.calcular_indice_debil(ae_azp['redcap'])

met.silhouette_score(ae_azp[['var1','var2']] ,ae_azp.redcap)


lqGlobal.calcular_indice_debil(ae_azp['ae_azpsa'])

met.silhouette_score(ae_azp.iloc[:,:2], ae_azp['ae_azp'])

