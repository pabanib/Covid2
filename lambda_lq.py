# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 19:18:49 2021

@author: Pablo
"""
import numpy as np
import pandas as pd
import elegir_modelo as em
import geopandas as gpd
import os
from lq import *


dir_principal = os.getcwd()
dir_datos = dir_principal+'\\datos'

covid = gpd.read_file(dir_datos+'/covid_periodos.shp', index = True)
covid = covid.set_index(['link','mes']).sort_index(level = 0)
covid = covid.loc[pd.IndexSlice[:,'2020-03':],:]
covid = covid.to_crs('POSGAR94')

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
covid_ult_mes = covid_acum.loc[pd.IndexSlice[:,'2021-07'],:]
covid_ult_mes = covid_ult_mes.reset_index(level = 'mes', drop = True)
geo = covid.loc[pd.IndexSlice[:,'2021-01'],'geometry']
geo = geo.reset_index(level = 'mes', drop = True)
codiprov = covid.loc[pd.IndexSlice[:,'2021-01'],['codpcia','departamen','provincia']]

import procesos
peri = procesos.peri_columna()

def datos_periodos(bd):
    gdf = gpd.GeoDataFrame(peri.fit_transform(bd))
    meses = []
    for i in list(gdf.columns):
        m = 'mes'+str(i)
        meses.append(m)
    gdf.columns = meses
    gdf = gdf.set_index(codiprov.index.get_level_values(0))
    gdf['personas'] = covid_ult_mes.personas
    gdf['geometry'] = geo
    
    return gdf
  

falle = datos_periodos(covid_acum['fallecido'])

grupos = gpd.read_file('G:/My Drive/Tesis/covid/covid2/df_R/positiv.shp')
grupos = grupos.set_index('link')

falle['sk'] = grupos['sk']

sk = falle.groupby('sk').sum()

s, l, lq_ = lq2(sk, 'mes16', 'personas')

    
#%%



r = region(falle.query("sk == 5"), 'personas','sk')        

l,lq_ = r.calc_lq()

r.evaluar_var()
r.evaluar_prom()


    
homogeneidad_particion(falle_dif, p)

falle_dif = falle.iloc[:,:17].T.diff().T
falle_dif['mes0'] = falle['mes0']
falle_dif['personas'] = falle.personas
falle_dif['sk'] = falle.sk

p = particion('sk')
p.transform(falle_dif)
falle_dif.sk.unique()


r = region(falle_dif.query("sk == 3"),'personas','sk')
r.evaluar_prom()
r.evaluar_var()
#%%

total = region(falle_dif, 'personas','sk')

total.evaluar_var()
total.evaluar_prom()

falle_prov = falle_dif.drop('sk', axis = 1)

falle_prov['pcia'] = codiprov['provincia'].values
prov = particion('pcia')

homogeneidad_particion(falle_prov, prov)

positiv = datos_periodos(covid_acum['clasificac'])
#positiv = positiv[['mes14']]
positiv = positiv.iloc[:,:17].T.diff().T
positiv = positiv.drop('mes0', axis = 1)
positiv['personas'] = falle.personas
positiv['sk'] = falle.sk

total = region(positiv, 'personas','sk')
total.evaluar_var()
total.evaluar_prom()

homogeneidad_particion(positiv, p)

pos_prov = positiv.drop('sk', axis = 1)
pos_prov['pcia'] = codiprov['provincia'].values

homogeneidad_particion(pos_prov, prov)

pos_aleat = positiv.drop('sk', axis = 1)
pos_aleat['clas'] = np.random.randint(0,6, len(pos_aleat))

clas = particion('clas')
homogeneidad_particion(pos_aleat, clas)

pos2021 = positiv[['mes9','mes10', 'mes11', 'mes12', 'mes13', 'mes14', 'mes15', 'mes16','personas']]

total  = region(pos2021, 'personas')
total.evaluar_prom()

pos2021['sk'] = positiv.sk.values

homogeneidad_particion(pos2021, p)
pos2021_prov = pos2021.drop('sk', axis = 1)
pos2021_prov['pcia'] = codiprov['provincia'].values
homogeneidad_particion(pos2021_prov, prov)
