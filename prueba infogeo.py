# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 11:56:03 2022

Este archivo agrega a los autoencoders la información espacial que provee el I de Moran local 

@author: Pablo
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import sklearn as sk
import sys
import os

# Ejecutar primero el archivo de lectura_datos.py en caso de que no exista covid_periodos.shp
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

#la variable se elige para comparar con diferentes opciones
variable = covid_acum[['clasificac','personas']]
#%% En esta sección se agregan las librerarías para calcular el I de Moran

# Además se crea una función que va distingir aquellos indices significativos

from libpysal.weights import Queen, Rook, KNN
covid_acum_geo = covid_ult_mes.copy()
covid_acum_geo['geometry'] = geo
w_queen = Queen.from_dataframe(covid_acum_geo)
w_rook = Rook.from_dataframe(covid_acum_geo)
w_knn = KNN.from_dataframe(covid_acum_geo, k = 6)


from esda.moran import Moran, Moran_Local
from splot.esda import moran_scatterplot

im = Moran(covid_acum_geo.clasificac, w_knn)
im.I
im.p_sim
im_local = Moran_Local(covid2.loc[pd.IndexSlice[:,'2021-01'],'clasificac'], w_knn)

covid2.loc[pd.IndexSlice[:,'2021-01'],'clasificac']

im_local.EI
moran_scatterplot(im_local, p=0.05)
moran_scatterplot(im, p=0.05)

im_local.Is.max()
im.I

im_local = Moran_Local(lq_[0].loc[pd.IndexSlice[:,'2021-01']], w_knn)

im_local.Is.max()

np.array(im_local.p_sim <0.05) + np.array(im_local.Is >0)

def dep_espacial(variable, matriz = w_knn):
   # Dada una variable esta función va arrojar una matriz de 2 columnas. La primera considera todos los I
   # de Moran positivos y significativos. La otra considera los I de Moran negativos y significativos
   # Es una matriz booleana.
    
   im = Moran_Local(variable, matriz)
   pos_sig = np.array(im.p_sim < 0.05)* np.array(im.Is > 0)
   neg_sig = np.array(im.p_sim < 0.05)* np.array(im.Is < 0)
   
   return np.c_[pos_sig, neg_sig]
   
dep_espacial(covid2.loc[pd.IndexSlice[:,'2021-01'],'personas']).sum(axis = 0)

#%% Esta sección sólo agrega las pipline necesarias para trabajar más rápido y agrega las métricas 
# que vamos a usar para elegir el mejor modelo


import sklearn.pipeline
import sklearn.preprocessing
from elegir_modelo import metodo, clustering_autoencoder,version
import procesos

pipe_coord = sk.pipeline.Pipeline([
    ('periodos', procesos.peri_columna()),
    ('coordenadas', procesos.agrega_centroides(centroides))
])

norm_l1 = sk.pipeline.Pipeline([
     ('std_scale', sk.preprocessing.StandardScaler())])#,
     #('norm_l1', sk.preprocessing.Normalizer('l1'))])
norm_l2 = sk.pipeline.Pipeline([
     ('std_scale', sk.preprocessing.StandardScaler()),
    ('norm_l2', sk.preprocessing.Normalizer('l2'))])

pipe = sk.pipeline.Pipeline([
    ('coord', pipe_coord),
    ('norml1', norm_l1)])    
    
# ddefinimos unas metricas para evaluar resultaods
from lq import *
lqcovid1 = lq_peri(covid_acum[['clasificac','personas']]/100)
lqcovid2 = lq_peri(covid_acum[['fallecido','personas']]/100)

from sklearn.metrics import silhouette_score,  calinski_harabasz_score
def siluetas(X, model):
    return silhouette_score(X, model.labels_)
def calinski(x, model):
    return sk.metrics.calinski_harabasz_score(x, model.labels_)
def inv_davies_bouldin(X,model):
    return (sk.metrics.davies_bouldin_score(X,model.labels_))**-1
def indice_lq(X, model):
    grupos = model.labels_
    ind = lqcovid1.calcular_indice_debil(grupos)
    ind2 = lqcovid2.calcular_indice_debil(grupos)
    return 1-((ind+ind2)/2)

import sklearn.cluster 
hiperparam = {
    'n_clusters': np.arange(5,30), 'init': ['k-means++', 'random'],
    'n_init' : [10]
}

metricas = {'sil': siluetas, 'cal': calinski, 'dav': inv_davies_bouldin, 'lqg': indice_lq}
metricas = {'lqg': indice_lq}

#%% 
"""
# El primer autoencoder que se va crear es considerando un I de moran de la variable poblacional "personas"
# La condición es si son positivos significativos o negativos significativos

ae = clustering_autoencoder(centroides, n_encoders = 8) #esta clase proviene del archivo elegir modelo.py

# Para traer los encoders usamos el siguiente método que usa la base de datos original y agrega como 
# variable de grupo los IM significativos

IM_var = dep_espacial(covid2.loc[pd.IndexSlice[:,'2021-01'],'personas']).astype(int)
IM_var
#Esta matriz de n*2 es la que determina la significatividad. Las significatividades varían de acuerdo a 
# la semilla
#EL método agregar grupo que es el utilizado para agregar esta info la va convertir en una matriz de n*4
#esto es porque genera dummies por cada variable (Revisar de cambiarlo después)
ae.agregar_grupo(IM_var).todense().shape

pobl = norm_l2.fit_transform(covid_acum_geo[['personas']])
#Realizamos unos encoder con la categoría de grupo provincias
#encoders_pcias = ae.fit_autoencoder_grupo(covid[covid_acum.columns],codiprov[['codpcia']].values )
encoders_pcias = ae.fit_autoencoder_grupo(covid[['clasificac','fallecido']],pobl,codiprov[['codpcia']].values )

encoders_pcias[:,:-2]
#Agragamos la información del I de Moran de la población

encoders_IMpob = ae.fit_autoencoder_grupo(covid[covid_acum.columns],pobl, IM_var)
encoders_IMpob[:,:-2]

# Realizamos los encoders con todos los I de Moran


p = procesos.peri_columna()

res = []
x = p.fit_transform(covid_acum[['clasificac','fallecido']])
for col in range(x.shape[1]):
    res.append(dep_espacial(x[:,col], w_knn).astype(int))

encoders = ae.fit_autoencoder_grupo(covid_acum,pobl,np.array(res).reshape(525,-1))
encoders[:,:-2]


# Utilizamos kmeans con cada uno de los encoders

#Primero vemos lo que es Kmeans de solo las variables geográficas

km_geo = metodo(sk.cluster.KMeans(), hiperparam, metricas)
km_geo.fit(ae.geo)
km_geo.best_metrics_

# Ahora aplicamos considerando cada uno de los encoders

ae_prov = metodo(sk.cluster.KMeans(), hiperparam, metricas)
ae_prov.fit(encoders_pcias)

ae_IMpob = metodo(sk.cluster.KMeans(), hiperparam, metricas)
ae_IMpob.fit(encoders_IMpob)

ae_IM = metodo(sk.cluster.KMeans(), hiperparam, metricas)
ae_IM.fit(encoders)

km_geo.best_metrics_
ae_prov.best_metrics_
ae_IMpob.best_metrics_
ae_IM.best_metrics_
ae_IM.fit(norm_l2.fit_transform(encoders))

ae_IM.best_metrics_
"""
def mapa_grupos(model):
    covid_acum_geo['grup'] = model.best_model_['modelo'].iloc[0].labels_
    gpd.GeoDataFrame(covid_acum_geo).plot(covid_acum_geo['grup']+1, figsize = (15,12), categorical = True, legend = True)
   

def metric_grup(model):
    std = sk.preprocessing.StandardScaler()
    for i in model.metrics.columns:
        met = model.metrics[i]
        met = std.fit_transform(met.values.reshape(-1,1))
        plt.scatter(model.modelos['modelo'].apply(lambda x: x.n_clusters) , met, label = i)
    plt.legend()
    plt.show()

"""
mapa_grupos(km_geo)
mapa_grupos(ae_prov)
mapa_grupos(ae_IMpob)
mapa_grupos(ae_IM)

metric_grup(ae_IM)
metric_grup(km_geo)

X = pipe.fit_transform(covid_acum)
X = np.c_[X,np.array(res).reshape(525,-1)]
X.shape

kmeans = metodo(sk.cluster.KMeans(), hiperparam, metricas)
kmeans.fit(X)

kmeans.best_metrics_
mapa_grupos(kmeans)

eva = evaluaciones_lq(ae_IM.best_model_['modelo'].iloc[0].labels_, ['fallecido','clasificac'],'personas')#.calcular_indices(covid_acum)
eva.calcular_indices(covid_acum)
eva.rdos_loc

eva2 = evaluaciones_lq(np.random.randint(0,5,525), ['fallecido','clasificac'],'personas')
eva2.calcular_indices(covid_acum)

eva.calcular_indices(covid_acum.loc[pd.IndexSlice[:,'2020-12'],:])



evaluaciones_lq(kmeans.best_model_['modelo'].iloc[0].labels_, ['fallecido','clasificac'],'personas').calcular_indices(covid_acum)

norm = sk.preprocessing.Normalizer('l2')
plt.plot(norm_l2.fit_transform(encoders))

plt.plot(encoders)
"""
#%%
"""
# Generamos datos random para ver cómo se comportan los índices

# generamos 525 variables poblacionales

pobla = np.random.randint(50000,100000, 525)
variables = []
for i in pobla:
    variables.append(np.random.randint(0,i,2))

variables = np.random.randint(0,5000,(525,2))
df_random = pd.DataFrame(np.c_[variables,pobla], index =covid2.loc[pd.IndexSlice[:,'2021-01'],'personas'].index, columns =['vari1','vari2','pobla'])

eva_random = evaluaciones_lq(codiprov['codpcia'].values, ['vari1','vari2'], 'pobla')
eva_random.calcular_indices(df_random)
eva_random.rdos_glob
eva_random.lq_locales(codiprov.codpcia.values)
eva_random.rdos_loc['vari1']

#= pd.DataFrame(np.random.randn(525,51).reshape(-1,3), index = covid_acum.index)

encoders_random = ae.fit_autoencoder(df_random, 2)

ae_random = metodo(sk.cluster.KMeans(), hiperparam, metricas)
ae_random.fit(norm_l2.fit_transform(encoders_random))
ae_random.best_metrics_
mapa_grupos(ae_random)

covid_acum_geo.mean()"""
#%%

# Agrupamos de manera aleatoria en el espacio

"""
eva_random2 = evaluaciones_lq(np.random.randint(0,15,525),['fallecido','clasificac'], 'personas')
eva_random2.calcular_indices(covid2/1)"""

#%% Simulación

centroides

pobl = norm_l1.fit_transform(covid_acum_geo[['personas']])
ag_cent = procesos.agrega_centroides(centroides)
xx = ag_cent.fit_transform(pobl)

# Utilizo como regiones dadas las provincias. Luego voy a simular los casos de acuerdo a ellas.
codiprov.codpcia
regiones = codiprov.codpcia.values

#covid_acum_geo.plot(regiones)
covid_acum_geo['personas'].groupby(regiones).count()
n_regiones = len(np.unique(regiones))

np.random.seed(5646)


personas = covid_acum_geo['personas']

def aleatorios(personas, regiones,geo):
    personas = pd.Series(personas)
    tot_pobl_region = personas.groupby(regiones).sum()
    n_regiones = len(np.unique(regiones))
    X_0 = np.random.randint(500,5000,n_regiones)
    
    Y = []
    for i in range(len(regiones)):
        cod = regiones[i]
        ind = np.where(np.unique(regiones) == cod)[0][0]
        TP = tot_pobl_region[cod]
        X = X_0[ind]
        theta = personas.iloc[i]/TP
        lamb = theta*X
        y =  np.random.poisson(lamb)
        Y.append(y)
    Y = np.array(Y).reshape(-1,1)
    columnas = []
    for i in range(Y.shape[1]):
        nombre = 'vari'+str(i+1)
        columnas.append(nombre)
    columnas.append('personas')
    
    df = pd.DataFrame(np.c_[Y,personas], columns = columnas )
    return gpd.GeoDataFrame(df, geometry = geo)


datos = aleatorios(personas,regiones,geo.values)

datos[lq(datos,'vari1','personas')[2] == lq(datos,'vari1','personas')[2].max()][['vari1','personas']]

plt.plot(lq(datos,'vari1','personas')[2])

lq_ = lq_multiple(datos[['vari1','personas']],{1:['vari1','personas']})
lq_.calcular_indice_debil(regiones)

X = norm_l1.fit_transform(datos[['vari1','personas']])

silhouette_score(X,regiones)

km = sk.cluster.KMeans(23) 
km.fit(X)
datos.plot(km.labels_)

silhouette_score(X,km.labels_)

lq_.calcular_indice(km.labels_)
lq_.calcular_indice(np.random.randint(0,15,525))

#datos['grup'] = aglo.labels_
#part = particion('grup')
#homogeneidad_particion(datos[['vari1','personas','grup']], part)

def homog(X, grupos):
    X = X.copy()
    X['grupo'] = grupos
    part = particion('grupo')
    #return part.transform(X)
    return homogeneidad_particion(X, part)

def homog_relat(X,var, grupos):
    X = X.copy()
    X['grupo'] = grupos
    part = particion('grupo')
    #return part.transform(X)
    return homogeneidad_particion(X, part)/np.var(X[var])

def homog_relat2(X,grupos):
    X = X.copy()
    X['grupo'] = grupos
    part = particion('grupo')
    HT = homog(X,np.ones(len(X)))
    return homogeneidad_particion(X, part)/HT

random_reg = np.random.randint(0,23,525)

homog(datos[['vari1','personas']], km.labels_)
homog(datos[['vari1','personas']], regiones)
homog(datos[['vari1','personas']], random_reg)
homog(datos[['vari1','personas']], np.ones(525)) #Homog. total

homog_relat(datos[['vari1','personas']],'vari1', np.ones(525))
homog_relat(datos[['vari1','personas']],'vari1', km.labels_)
homog_relat(datos[['vari1','personas']],'vari1', regiones)
homog_relat(datos[['vari1','personas']],'vari1', random_reg)

homog_relat2(datos[['vari1','personas']], np.ones(525))
homog_relat2(datos[['vari1','personas']], km.labels_)
homog_relat2(datos[['vari1','personas']], regiones)
homog_relat2(datos[['vari1','personas']], random_reg)


from sklearn import metrics

metrics.adjusted_rand_score(regiones,km.labels_)
metrics.adjusted_rand_score(regiones,random_reg)
metrics.adjusted_rand_score(regiones,regiones)

metrics.adjusted_mutual_info_score(regiones,km.labels_)
metrics.adjusted_mutual_info_score(regiones,random_reg)
metrics.adjusted_mutual_info_score(regiones,regiones)

def metricas_region(datos,regiones, reg_predic):
    lq_ = lq_multiple(datos[['vari1','personas']],{1:['vari1','personas']})  
    ilq = lq_.calcular_indice_debil(reg_predic)
    hg = homog(datos[['vari1','personas']],reg_predic)
    hgr = homog_relat(datos[['vari1','personas']],'vari1',reg_predic)
    hgr2 = homog_relat2(datos[['vari1','personas']],reg_predic) 
    sil = silhouette_score(stand.fit_transform(datos[['vari1','personas']]), reg_predic)
    ars = metrics.adjusted_rand_score(regiones, reg_predic)
    mis = metrics.adjusted_mutual_info_score(regiones, reg_predic)

    return [ilq,hg,hgr,hgr2,sil,ars,mis]

stand = sk.preprocessing.StandardScaler()
"""
metricas_region(datos,regiones,regiones)
metricas_region(datos,regiones,km.labels_)
metricas_region(datos,regiones,random_reg)

np.random.seed(496846)
df = []
i = 0
while i < 100:
    personas = np.random.randint(5000,50000,525)
    datos = aleatorios(personas,regiones,geo.values)
    res = metricas_region(datos,regiones, np.random.randint(0,24,525))
    
    df.append(res)
    i += 1

df = pd.DataFrame(df)

df[[0,2,3]].plot()


datos = aleatorios(covid_acum_geo.personas,regiones,geo.values)

np.random.seed(6846)
df = []
i = 0
while i < 100:
    n_grupos = i+10
    km = sk.cluster.AgglomerativeClustering(n_clusters = n_grupos)
    #km = sk.cluster.KMeans(n_grupos)
    km.fit(stand.fit_transform(datos[['vari1','personas']]))
    regiones_ = km.labels_

    res = metricas_region(datos,regiones, regiones_)    
    res.append(n_grupos)
    df.append(res)
    i += 1

df = pd.DataFrame(df, columns = ['ilq','hg','hgr','sil','ars','mis','grupos'])

df[['ilq','hgr','mis','sil']].plot()
df
"""
#%%

n = 20
lamb = 50 
X_i = np.random.poisson(lamb)
t_n = np.random.randint(250,2000,n)
T_i = t_n.sum()

lambdas = np.round(X_i * t_n/T_i)

X = []
E = []
for i in range(1000):
    x = []
    ee = []
    for j in lambdas:
        xx = np.random.poisson(j)
        e = xx-j        
        x.append(xx)
        ee.append(e)
    X.append(np.array(x))
    E.append(np.array(ee))
X = np.array(X)
np.sum(X.sum(axis = 1) == X_i)

plt.hist(E[0])

#%%

from numpy import exp
from scipy.integrate import odeint

class casos_region():
    
    def __init__(self, pop, i_0,gamma, sigma, peris = 17):
        
        self.pop_size = pop
        self.gamma = gamma
        self.sigma = sigma
        
        # Condiciones iniciales 
        i_0 = i_0/pop
        e_0 = 4*i_0
        s_0 = 1-i_0-e_0
        
        self.x_0 = s_0,e_0,i_0
        
        R0 = 1+np.random.rand()*2
        time = range(peris)
        self.i_path, self.c_path = self.solve_path(R0, time,self.x_0)
                
    
    def F(self,x,t,R0 = 1.6):
        
        s,e,i = x
        gamma = self.gamma
        sigma = self.sigma
        
        # Nueva exposición de susceptibles
        beta = R0(t)*gamma if callable(R0) else R0*gamma
        ne = beta*s*i
        
        # Derivadas en el tiempo
        ds = -ne
        de = ne- sigma*e
        di = sigma*e - gamma*i
        
        return ds, de, di
    
      
    def solve_path(self,R0, t_vec, x_init):
        
        G = lambda x, t: self.F(x,t, R0)
        s_path, e_path, i_path = odeint(G, x_init, t_vec).transpose()
        
        c_path = 1-s_path- e_path
        if np.any(i_path < 0):
            i_path[i_path < 0] = 0.000001
        elif np.any(i_path == 0):
            i_path[i_path == 0] = 0.000001
                    
        return i_path, c_path
    
    def ploting(self):
        
        plt.plot(self.i_path)
        plt.plot(self.c_path)
    
personas = covid_acum_geo.personas.groupby(regiones).sum()

class area_region():
    
    def __init__(self,personas, regiones,peris = 17,geo = geo):
        self.personas =pd.Series(personas)
        self.regiones = regiones
        self.peris = peris
        self.geo = geo
        self.tot_pobl_region = personas.groupby(regiones).sum()
                

    def proceso_temp_region(self):
        l_reg = []
        casos  = []
        for i in self.tot_pobl_region:
            pop = i 
            x_0 = np.random.randint(0,pop*0.0005)
            gamma = 1+np.random.rand()*5
            sigma = 1+np.random.rand()*9
            cr = casos_region(pop, x_0, gamma, sigma, self.peris)
            l_reg.append(cr)
            casos.append(cr.i_path*pop)
        self.l_reg = l_reg    
        casos = pd.DataFrame(casos, index = self.tot_pobl_region.index)

        return casos
    
    def proceso_area(self):
        personas = self.personas
        tot_pobl_region = self.tot_pobl_region 
        regiones = self.regiones
        n_regiones = len(np.unique(regiones))
       
        casos = self.proceso_temp_region()
        self.casos = casos
        
        Y = []
        Dic = {}
        for j in casos.columns:
            X_0 = casos.loc[:,j] 
            dic = {}
            YY = []
            for i in range(len(regiones)):
                cod = regiones[i]
                ind = np.where(np.unique(regiones) == cod)[0][0]
                TP = tot_pobl_region[cod]
                X = X_0.iloc[ind]
                theta = personas.iloc[i]/TP
                lamb = theta*X
                y =  np.random.poisson(lamb)
                YY.append(y)
                dic[i] = [cod,y,X,personas.iloc[i],TP]
            Y.append(YY)
            Dic[j] = dic
        Y = np.array(Y).T#.reshape(len(personas), len(casos.columns))
        columnas = []
        self.Dic = Dic
        for i in range(Y.shape[1]):
            nombre = 'peri'+str(i+1)
            columnas.append(nombre)
        columnas.append('personas')
            
        df = pd.DataFrame(np.c_[Y,personas], columns = columnas )
        self.area_casos = df
        return gpd.GeoDataFrame(df, geometry = self.geo.values)
    
    def convertir_a_panel(self):
        
        df_ = df[df.columns[:list(df.columns).index('personas')]].stack()
        p = np.array([[self.personas.values,]]*self.peris)
        
        X = np.c_[df_.values, p.T.reshape(-1,1)]
        return pd.DataFrame(X, index = df_.index, columns = ['vari1','personas'])
                

ar = area_region(covid_acum_geo.personas, regiones,17)

df = ar.proceso_area()
df.loc[520]
regiones[520]
ar.casos.loc['90']
df.groupby(regiones).sum()

df
panel_df = ar.convertir_a_panel()
panel_df

eva = evaluaciones_lq(regiones, ['vari1'], 'personas')
eva.calcular_indices(panel_df)
eva2 = evaluaciones_lq(np.random.randint(0,24,525), ['vari1'],'personas')
eva2.calcular_indices(panel_df)

homog_relat2(df[['peri1','personas']], regiones)
homog_relat2(df[['peri1','personas']], np.random.randint(0,24,525))

eva.rdos_loc['vari1']['regiones']['94']['region'].lqs

df[regiones == '94']

peris = df.columns[:list(df.columns).index('geometry')]
km = sk.cluster.KMeans(24)
km.fit(pipe.fit_transform(panel_df))
random_reg = np.random.randint(0,24,525)
ind_lq = lq_peri(panel_df)


homog(df[peris], regiones)
homog_relat2(df[peris],regiones)
ind_lq.calcular_indice_debil(regiones)
sk.metrics.silhouette_score(pipe.fit_transform(panel_df), regiones)

homog(df,km.labels_)
homog_relat2(df,km.labels_)
ind_lq.calcular_indice_debil(km.labels_)
sk.metrics.silhouette_score(pipe.fit_transform(panel_df), km.labels_)

homog(df[peris],random_reg)
homog_relat2(df[peris],random_reg)
ind_lq.calcular_indice_debil(random_reg)
sk.metrics.silhouette_score(pipe.fit_transform(panel_df), random_reg)

#%% Probando algoritmos

def Hg_relat(X,model):
    
    return 1-homog_relat2(df[peris],model.labels_)

def MI(X,model):
    real = regiones
    return metrics.adjusted_mutual_info_score(real,model.labels_)

hiperparam['n_clusters'] = np.arange(5,50)
hiperparam
metricas['Hgr'] = Hg_relat

km = metodo(sk.cluster.KMeans(), hiperparam, {'mi': MI})
km.fit(pipe.fit_transform(panel_df))
km.best_metrics_
km.best_model_

def compara_metricas(modelo, datos):
    try:
        data = pipe.fit_transform(datos)
    except:
        data = datos
    result = []
    for m in range(len(modelo.modelos)):
        model = modelo.modelos.iloc[m]['modelo']
        mi = MI(data,model)
        ilq = indice_lq(data, model)
        hgr = Hg_relat(data,model)
        sil = siluetas(data, model)
        cal = calinski(data, model)
        dav = inv_davies_bouldin(data,model)
        iner = model.inertia_
        result.append(np.array([mi,ilq,hgr,sil,cal,dav,iner]))
    result = pd.DataFrame(result, columns = ['mi','ilq','hgr','sil','cal','dav','iner'])
    return result

result = compara_metricas(km, panel_df)
result.corr()

result[['mi','ilq','hgr','sil']].plot()
result[result.hgr == result.hgr.min()]

result[['mi','ilq','hgr','sil']].drop(index = 25).plot()
result.drop(index = 25).corr()

homog_relat2(df,km.best_model_['modelo'].iloc[0].labels_ )

#km.metrics.Hgr.max()
km.modelos.iloc[13]['modelo'].inertia_
mapa_grupos(km)

homog_relat2(df, km.modelos.iloc[13]['modelo'].labels_)

df.plot(km.modelos.iloc[13]['modelo'].labels_)

ae = clustering_autoencoder(centroides, n_encoders = 8)
vl = ae.fit_autoencoder(panel_df, 1)

ae_km = copy(km)
ae_km.fit(vl)
ae_km.best_metrics_

result2 = compara_metricas(ae_km,vl)
result2.corr()
result2[['mi','ilq','hgr','sil']].plot()

new_panel = panel_df['vari1']/panel_df['personas']
new_panel = pd.DataFrame(new_panel)
vl2 = ae.fit_autoencoder(new_panel, 1)
ae_km.fit(vl2)
ae_km.best_metrics_

result3 = compara_metricas(ae_km,vl2)
result3.corr()
result3[['mi','ilq','hgr','sil']].plot()
pipe.fit_transform(new_panel)
