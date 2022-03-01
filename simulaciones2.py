# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 08:26:31 2022

@author: Pablo
"""

from lectura_datos import *

covid.plot('personas')

codiprov.groupby('provincia').describe()

regiones = {'noroeste' : ('Jujuy','Salta','Tucumán','Santiago del Estero','Catamarca'),
            'noreste'  : ('Formosa', 'Chaco', 'Misiones','Corrientes','Santa Fe','Entre Ríos'),
            'cuyo' : ('La Rioja','San Juan','Mendoza','San Luis'),
            'pampeana' : ('Córdoba', 'La Pampa', 'Buenos Aires','Ciudad Autónoma de Buenos Aires' ),
            'patagonia' : ('Neuquén', 'Río Negro', 'Chubut', 'Santa Cruz', 'Tierra del Fuego')    
    }


lista_regiones = []
errores = []
for i in range(len(codiprov)):
    prov = codiprov.iloc[i].provincia
    j = 1
    pertenece = False
    for k in regiones.keys():
        l = regiones[k]
        if prov in l:
            lista_regiones.append(j)
            pertenece = True
        else:
        
            pass
        j += 1
    if pertenece == False:
        errores.append(codiprov.iloc[i]) 


cov =  covid.loc[pd.IndexSlice[:,'2021-01'],:]
cov['reg'] = lista_regiones
cov.plot('reg')

var = cov.reg.map({1:0.01,2:0.05,3:0.1, 4:0.15, 5:0.2})* cov.personas


from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import elegir_modelo as em

param = {'n_clusters' : [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}

def siluetas(X, model):
    return silhouette_score(X, model.labels_)

metricas = {'sil': siluetas}

km = em.metodo(KMeans(),param,metricas)
km.fit(var.values.reshape(-1,1))

km.best_metrics_
km.best_model_
km.metrics

def mapa_grupos(data,model):
    data['grup'] = model.best_model_['modelo'].iloc[0].labels_
    gpd.GeoDataFrame(data).plot(data['grup']+1, figsize = (15,12), categorical = True, legend = True)
   
mapa_grupos(cov,km)

km.fit((var/cov.personas).values.reshape(-1,1))
mapa_grupos(cov,km)

km.best_metrics_

aglo = em.metodo(AgglomerativeClustering(), param, metricas)

aglo.fit((var/cov.personas).values.reshape(-1,1))
mapa_grupos(cov,aglo)
aglo.best_metrics_


