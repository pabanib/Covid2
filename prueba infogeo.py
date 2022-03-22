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
#covid = covid.to_crs({'init':'POSGAR94'})

# Separamos los campos geometricos del dataframe
geo = covid.loc[pd.IndexSlice[:,'2021-01'],'geometry']
geo = geo.reset_index(level = 'mes', drop = True)
centroides = covid.loc[pd.IndexSlice[:,'2021-01'],'geometry'].centroid#to_crs('POSGAR94').centroid
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

#%% Simulación de una sola variable en un solo periodo

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
datos
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

def homog(X, grupos,poblacion = 'personas'):
    # X debe ser un data frame que contenga las columnas de la variable y la poblaional
    X = X.copy()
    X['grupo'] = grupos
    part = particion('grupo')
    #return part.transform(X)
    return homogeneidad_particion(X, part,poblacion)

def homog_relat(X,var, grupos):
    # X debe ser un data frame que contenga las columnas de la variable y la poblaional
    X = X.copy()
    X['grupo'] = grupos
    part = particion('grupo')
    #return part.transform(X)
    return homogeneidad_particion(X, part)/np.var(X[var])

def homog_relat2(X,grupos,poblacion = 'personas'):
    # X debe ser un data frame que contenga las columnas de la variable y la poblaional
    X = X.copy()
    X['grupo'] = grupos
    part = particion('grupo')
    HT = homog(X,np.ones(len(X)), poblacion)
    return homogeneidad_particion(X, part, poblacion)/HT

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

#%% Simulaciones en el tiempo modelo SIR. 

from numpy import exp
from scipy.integrate import odeint

class casos_region():
    
    def __init__(self, pop, i_0,gamma, sigma, peris = 17):
        
        self.pop_size = pop
        self.gamma = gamma
        self.sigma = sigma
        self.peris = peris
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
    
#Trabajamos con la cantidad de personas por cada región fija
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
                lamb = round(theta*X + np.random.randn()*theta*X,0)
                lamb = lamb if lamb>0 else 0.00001
                y =  np.random.poisson(lamb)+np.random.randint(0,round(lamb,0)+3)
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
    
    def proceso_area_mult(self, variables):
        personas = self.personas
        tot_pobl_region = self.tot_pobl_region 
        regiones = self.regiones
        n_regiones = len(np.unique(regiones))

        casos = self.proceso_temp_region()
        for i in range(variables-1):
            casos_i = self.proceso_temp_region()
            casos = pd.merge(casos,casos_i,left_index = True,right_index = True)
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
                lamb = round(theta*X + np.random.randn()*theta*X,0)
                lamb = lamb if lamb>0 else 0.00001
                y =  np.random.poisson(lamb)+np.random.randint(0,round(lamb,0)+3)
                YY.append(y)
                dic[i] = [cod,y,X,personas.iloc[i],TP]
            Y.append(YY)
            Dic[j] = dic
        Y = np.array(Y).T#.reshape(len(personas), len(casos.columns))
        columnas = []
        self.Dic = Dic
        for j in range(variables):
            vari = 'vari'+str(j)
            for i in range(self.peris):
                nombre = (vari,'peri'+str(i+1))
                columnas.append(nombre)
        columnas = tuple(columnas)
        columnas = pd.MultiIndex.from_tuples(columnas, names = ['variables','periodos'])
        self.columnas = columnas
        #columnas.append('personas')
        df = pd.DataFrame(Y, columns = columnas )
        df['personas'] = personas.reset_index(drop = True)
        self.area_casos = df
        return gpd.GeoDataFrame(df, geometry = self.geo.values)
   
    def convertir_a_panel(self,variables = 1):
        
        df = self.area_casos
        if variables > 1:
            df_ = df[df.columns[:list(df.columns).index(('personas',''))]].stack()
            columnas = list(df_.columns.get_level_values(0).unique())
        else:
            df_ = df[df.columns[:list(df.columns).index('personas')]].stack()
            columnas = list(['vari1'])
        columnas.append('personas')
        p = np.array([[self.personas.values,]]*self.peris)
        
        X = np.c_[df_.values, p.T.reshape(-1,1)]
        return pd.DataFrame(X, index = df_.index, columns = columnas)
#%%
             

# se simula el caso de una sola variable en varios periodos
np.random.seed(32)

ar = area_region(covid_acum_geo.personas, regiones,17)
df = ar.proceso_area()
df
# convertir a panel estructura los datos en forma de panel para poder trabajarlos
panel_df = ar.convertir_a_panel()
panel_df

# Las evaluaciones lq son las que calculan cuanto es la diferencia cuadrática y cuanto es el índice lq
eva = evaluaciones_lq(regiones, ['vari1'], 'personas')
eva.calcular_indices(panel_df)
eva2 = evaluaciones_lq(np.random.randint(0,24,525), ['vari1'],'personas')
eva2.calcular_indices(panel_df)
# Se puede ver que el índice lq falla cuando la agrupación es random, lo cual no debería suceder

# La homogeneidad dividida la homogeneidad total muestra claramanete que una agrupación es la correcta y 
# la otra es random
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

class proceso_aleatorio():
    def __init__(self, personas, regiones,variables = 1 ,periodos = 17):
        self.personas = personas
        self.regiones = regiones
        self.variables = variables
        self.periodos = periodos
    
    def generar_datos(self, semilla = 32):
        personas = self.personas
        regiones = self.regiones
        variables = self.variables
        periodos = self.periodos
        np.random.seed(semilla) 
        ar = area_region(personas, regiones,periodos)
        if variables == 1:
            self.df = ar.proceso_area()
        elif variables > 1:
            self.df = ar.proceso_area_mult(variables)
            
        self.panel_df = ar.convertir_a_panel(variables)
        self.new_panel = self.panel_df.apply(lambda x: x/self.panel_df.personas)
        self.new_panel = self.new_panel.drop('personas', axis = 1)
        self.centroides = centroides
        self.inputs, self.outputs = self.__genera_inputs()
        self.inputs_moran = copy(self.inputs)
        self.inputs_moran.append(self.__genera_I_moran())
        
        return self
    def __genera_inputs(self):
        inputs = []
        for v in self.new_panel.columns:
            in1 = pipe.fit_transform(self.new_panel[[v]])[:,:-2]
            inputs.append(in1)
        
        outputs = pipe.fit_transform(self.new_panel)[:,:-2]
        return (inputs,outputs)
    
    def __genera_I_moran(self, W = w_knn):
        df = self.df
        I_locales = []
        for c in df.columns[:-1]:
            I = Moran_Local(df[c], W).Is
            I_locales.append(I)

        I_locales = norm_l2.fit_transform(np.array(I_locales)).T
        
        return I_locales

#pa = proceso_aleatorio(covid_acum_geo.personas,regiones, variables =2).generar_datos()


#%% Probando algoritmos
class calcular_incicadores():
    def __init__(self, df,panel_Df ,regiones = regiones):
        self.df = df
        self.pobl = self.df.columns[-2]
        self.regiones = regiones
        self.lqperis = []
        for c in panel_df.columns[:-1]:
            lq = lq_peri(panel_df[[c,panel_df.columns[-1]]])
            self.lqperis.append(copy(lq))
    
    def indice_lq(self, X, model):
        grupos = model.labels_
        ind = 0
        for i in self.lqperis:
            ind += i.calcular_indice_debil(grupos)
       
        return 1-((ind)/len(self.lqperis))
   
    
    def Hg_relat(self,X,model):
    
        return 1-homog_relat2(self.df,model.labels_, self.pobl)

    def MI(self,X,model):
        real = self.regiones
        return metrics.adjusted_mutual_info_score(real,model.labels_)

    def compara_metricas(self,modelo, datos):
        try:
            data = pipe.fit_transform(datos)
        except:
            data = datos
        result = []
        for m in range(len(modelo.modelos)):
            model = modelo.modelos.iloc[m]['modelo']
            mi = self.MI(data,model)
            ilq = self.indice_lq(data, model)
            hgr = self.Hg_relat(data, model)
            sil = siluetas(data, model)
            cal = calinski(data, model)
            dav = inv_davies_bouldin(data,model)
            try:
                iner = model.inertia_
            except:
                iner = np.nan
            result.append(np.array([mi,ilq,hgr,sil,cal,dav,iner]))
        result = pd.DataFrame(result, columns = ['mi','ilq','hgr','sil','cal','dav','iner'])
        return result

# transformación de variables a relación de casos con personas
new_panel = panel_df['vari1']/panel_df['personas']
new_panel = pd.DataFrame(new_panel)

hiperparam['n_clusters'] = np.arange(5,50)
hiperparam
hiperparam_aglo = {'n_clusters': np.arange(5,50),
                   'connectivity': [w_knn.sparse],
 'affinity': ['l1', 'l2', 'manhattan', 'cosine'],
 'linkage': ['complete', 'average', 'single']}

indicadores = calcular_incicadores(df, panel_df)


#km con población por separado ponderanda igual que otras variables
#km = metodo(sk.cluster.KMeans(), hiperparam, {'mi': indicadores.MI})
km = metodo(sk.cluster.KMeans(), hiperparam, {'lqg': indicadores.indice_lq, 'hgr': indicadores.Hg_relat})
km.fit(pipe.fit_transform(panel_df))
km.best_metrics_
km.best_model_

df.plot(km.modelos[km.metrics.hgr == km.metrics.hgr.min()]['modelo'].iloc[0].labels_, categorical = True)

result = indicadores.compara_metricas(km, panel_df)
result.corr()

result[['mi','ilq','hgr','sil']].plot()
min_hgr = result[result.hgr == result.hgr.min()]

result[['mi','ilq','sil']].drop(index = min_hgr.index[0]).plot()
result.drop(index = min_hgr.index[0]).corr()

km.modelos.iloc[min_hgr.index[0]]

homog_relat2(df,km.best_model_['modelo'].iloc[0].labels_ )

km.modelos.iloc[13]['modelo'].inertia_
mapa_grupos(km)
homog_relat2(df, km.modelos.iloc[13]['modelo'].labels_)

# km con relación entre casos sobre poblacion
km2 = copy(km)
km2.fit(pipe.fit_transform(new_panel))

km2.best_metrics_
km2.best_model_
indicadores.MI(df, km2.best_model_['modelo'].iloc[0])

mapa_grupos(km2)
result_km2 = indicadores.compara_metricas(km2, new_panel)
result_km2.corr()
result_km2[['mi','ilq','hgr','sil']].plot()
result_km2.describe()

#Jerárquico considerando casos sobre poblacion
aglo = metodo(sk.cluster.AgglomerativeClustering(), hiperparam_aglo,{'lqg': indicadores.indice_lq, 'hgr': indicadores.Hg_relat})
aglo.fit(pipe.fit_transform(new_panel))
aglo.best_metrics_
mapa_grupos(aglo)
indicadores.MI(df, aglo.best_model_['modelo'].iloc[0])
result_aglo = indicadores.compara_metricas(aglo,new_panel)

# kmeans considerando variables latentes partiendo con poblacion separada

ae = clustering_autoencoder(centroides, n_encoders = 8)
vl = ae.fit_autoencoder(panel_df, 1)

ae_km = copy(km)
ae_km.fit(vl)
ae_km.best_metrics_

result2 = indicadores.compara_metricas(ae_km,vl)
result2.corr()
result2[['mi','ilq','hgr','sil']].plot()
result2.describe()

# autoencoders kmeans considerando casos sobre población
vl2 = ae.fit_autoencoder(new_panel, 1)
ae_km.fit(vl2)
ae_km.best_metrics_

result3 = indicadores.compara_metricas(ae_km,vl2)
result3.corr()
result3[['mi','ilq','hgr','sil']].plot()
pipe.fit_transform(new_panel)
mapa_grupos(ae_km)
indicadores.MI(df, ae_km.best_model_['modelo'].iloc[0])

result3.describe()
# autoencoders considerando variables geográficas

Moran(df['personas'], w_knn).I
Moran_Local(df['peri2'], w_knn).Is

I_locales = []
for c in df.columns[:list(df.columns).index('geometry')]:
    I = Moran_Local(df[c], w_knn).Is
    I_locales.append(I)

I_locales = np.array(I_locales)

norm_l2.fit_transform(I_locales).T

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

n_encoders = 8
input1 = layers.Input(shape = [17,])
input2 = layers.Input(shape= [18,])
encoder1 = layers.Dense(n_encoders, activation = "relu", kernel_regularizer = regularizers.l1(0.1))(input1)
encoder2 = layers.Dense(n_encoders, activation = "relu", kernel_regularizer = regularizers.l1(0.1))(input2)
concat = layers.concatenate([encoder1,encoder2])
encoder = layers.Dense(n_encoders, activation = "relu")(concat)
      
decoder = layers.Dense(17+18, activation = "sigmoid")(encoder)
autoencoder = Model(inputs = [input1,input2], outputs = decoder)
autoencoder.compile(optimizer = "sgd", loss = "mse")# "categorical_crossentropy")
enco = Model(inputs = [input1,input2], outputs = encoder)

in1 = pipe.fit_transform(new_panel)[:,:-2]
in2 = norm_l2.fit_transform(I_locales).T
X = np.c_[in1,in2]

autoencoder.fit((in1,in2),X,epochs = 50)

vl =  np.c_[enco.predict((in1,in2)),pipe.fit_transform(new_panel)[:,-2:]]

ae_km_moran = copy(ae_km)
ae_km_moran.fit(vl)
ae_km_moran.best_metrics_
ae_km_moran.best_model_
mapa_grupos(ae_km_moran)

result_aemoran = indicadores.compara_metricas(ae_km_moran, new_panel)
result_aemoran.corr()

#### km2 agregando indices de moran
xx = np.c_[X,pipe.fit_transform(new_panel)[:,-2:]]
xx = norm_l2.fit_transform(xx)
km3 = copy(km)
km3.fit(xx)
km3.best_metrics_
mapa_grupos(km3)
result_km3 = indicadores.compara_metricas(km3,new_panel)


###### Resultados de las diferentes metodologías

km.best_metrics_
result.iloc[km.best_metrics_.index] 
result.mean()
result.std()

km2.best_metrics_
result2.iloc[km2.best_metrics_.index] 
result2.mean()
result2.std()

aglo.best_metrics_
result_aglo.iloc[aglo.best_metrics_.index]
result_aglo.mean()
result_aglo.std()
mapa_grupos(aglo)


ae_km.best_metrics_
result3.iloc[ae_km.best_metrics_.index] 
result3.mean()
result3.std()

ae_km_moran.best_metrics_
result_aemoran.iloc[ae_km_moran.best_metrics_.index] 
result_aemoran.mean()
result_aemoran.std()

km3.best_metrics_
result_km3.iloc[km3.best_metrics_.index] 
result_km3.mean()
result_km3.std()


##### modelo real

eva = evaluaciones_lq(regiones, ['vari1'], 'personas')
eva.ajustar_datos(panel_df)
eva.calcular_indices(panel_df)
homog_relat2(df, regiones)
silhouette_score(pipe.fit_transform(panel_df), regiones)
calinski_harabasz_score(pipe.fit_transform(panel_df), regiones)
sk.metrics.davies_bouldin_score(pipe.fit_transform(panel_df), regiones)

#### ---###


class agrupamientos():
    def __init__(self, proceso_alaetorio,metodos):
        # proceso_aleatorio es el objeto proceso aleatorio
        # metodos es una lista de objetos tipo elegir_modelo.metodo
        self.pa = proceso_alaetorio
        self.metodos = metodos
    
    def variables_latentes(self, autoencoder, encoder):
        # autoencoder debe ser un objeto de keras.Model y compilado
        # encoder tambies un objeto tipo keras.Model
            
        autoencoder.fit(self.pa.inputs_moran, self.pa.outputs, epochs = 50)
        vl = np.c_[encoder.predict(self.pa.inputs_moran), pipe.fit_transform(self.pa.new_panel)[:,-2:]]
        return vl      
        
    def clustering(self,autoencoder,encoder):
        vl_moran = self.variables_latentes(autoencoder, encoder)
        ae = clustering_autoencoder(centroides, 8)
        vl = ae.fit_autoencoder(self.pa.new_panel, self.pa.variables)
        
        rdos = []
        self.modelos = []
        indicadores = {}
        j = 1
        for m in self.metodos:
            for data in [self.pa.panel_df, self.pa.new_panel]:
                m.fit(pipe.fit_transform(data))
                self.modelos.append(copy(m))
                indicadores[j] = self.__indicadores__(data, m)                
                rdos.append(m.best_metrics_.iloc[0][0])
                j += 1
            for data in  [vl, vl_moran]:
                m.fit(data)
                self.modelos.append(copy(m))
                indicadores[j] = self.__indicadores__(data, m)
                rdos.append(m.best_metrics_.iloc[0][0])
                j += 1
        self.indicadores = indicadores
        return rdos
    
    def __indicadores__(self, data, metodo):
        
        ind = calcular_incicadores(self.pa.df, self.pa.panel_df, self.pa.regiones)
        return ind.compara_metricas(metodo, data)
"""
pa = proceso_aleatorio(covid_acum_geo.personas, regiones).generar_datos()

agru = agrupamientos(pa,[km])
agru.clustering(autoencoder, enco)

rdos
rdos = []
for i in np.random.randint(20,95645,10):
    pa = proceso_aleatorio(covid_acum_geo.personas, regiones, variables = 1).generar_datos(i)
    agru = agrupamientos(pa,[km,aglo])
    r = agru.clustering(autoencoder, enco)
    rdos.append(np.array(r))
np.array(rdos)

import pickle 

with open("Resultados/rdos_mi.pickle", "wb") as f:
    pickle.dump(rdos, f)

"""

#%%
# simulaciones con más de una variable

import tslearn.clustering as ts

arm = area_region(covid_acum_geo.personas,regiones)
df2 = arm.proceso_area_mult(2)
panel_df2 = arm.convertir_a_panel(2)
panel_df2

new_panel2 = panel_df2.apply(lambda x: x/panel_df2.personas)
new_panel2 = new_panel2.drop('personas', axis = 1)

ind = calcular_incicadores(df2,panel_df2, regiones)
km = metodo(sk.cluster.KMeans(), hiperparam, {'mi': ind.MI})

# primero aplicamos kmeans
km.fit(pipe.fit_transform(new_panel2))
km.best_metrics_
mapa_grupos(km)
#new_panel2['vari3'] = np.random.rand(8925)
metrics1 = ind.compara_metricas(km, new_panel2)
metrics1.corr()
metrics1[['hgr','mi']].plot()
km.modelos[metrics1.hgr == metrics1.hgr.max()]
km.best_model_

df2.plot(km.modelos[metrics1.hgr == metrics1.hgr.max()]['modelo'].iloc[0].labels_, categorical = True)
df2.plot(km.modelos[metrics1.hgr == metrics1.hgr.min()]['modelo'].iloc[0].labels_,categorical = True)
mapa_grupos(km)

# Aplicamos el metodo jerárquico 
aglo = metodo(sk.cluster.AgglomerativeClustering(), hiperparam_aglo, {'mi': ind.MI})

aglo.fit(pipe.fit_transform(new_panel2))
aglo.best_metrics_
mapa_grupos(aglo)
aglo.best_model_

metrics0 = ind.compara_metricas(aglo, new_panel2)
ind.indice_lq(new_panel2, aglo.best_model_['modelo'].iloc[0])
ind.Hg_relat(new_panel2, aglo.best_model_['modelo'].iloc[0])

# aplicamos autoencoders considerando ambas variables por separado
n_encoders = 8
input1 = layers.Input(shape = [17,])
input2 = layers.Input(shape= [17,])
#input3 = layers.Input(shape= [17,])
#input4 = layers.Input(shape= [17,])
encoder1 = layers.Dense(n_encoders, activation = "relu", kernel_regularizer = regularizers.l1(0.1))(input1)
encoder2 = layers.Dense(n_encoders, activation = "relu", kernel_regularizer = regularizers.l1(0.1))(input2)
#encoder3 = layers.Dense(n_encoders, activation = "relu", kernel_regularizer = regularizers.l1(0.1))(input3)
#encoder4 = layers.Dense(n_encoders, activation = "relu", kernel_regularizer = regularizers.l1(0.1))(input4)
concat = layers.concatenate([encoder1,encoder2])
encoder = layers.Dense(n_encoders, activation = "relu")(concat)
      
decoder = layers.Dense(17*2, activation = "sigmoid")(encoder)
autoencoder = Model(inputs = [input1,input2], outputs = decoder)
autoencoder.compile(optimizer = "sgd", loss = "mse")#"categorical_crossentropy")
enco = Model(inputs = [input1,input2], outputs = encoder)

inputs = []
for v in new_panel2.columns:
    in1 = pipe.fit_transform(new_panel2[[v]])[:,:-2]
    inputs.append(in1)
X = np.array(inputs).reshape(525,2*17)

autoencoder.fit(inputs, X, epochs = 50)
enco.predict(inputs)

vl = np.c_[enco.predict(inputs),pipe.fit_transform(new_panel2[[v]])[:,-2:] ]
ae_km = copy(km)
ae_km.fit(vl)
ae_km.best_metrics_
mapa_grupos(ae_km)
metrics2 = ind.compara_metricas(ae_km, new_panel2)
metrics2.corr()

grupos = []
for i in range(len(ae_km.modelos)):
    g = ae_km.modelos['modelo'].iloc[i].n_clusters
    grupos.append(g)
metrics2['grupos'] = grupos

# autoencoders con I_locales
n_encoders = 2
input1 = layers.Input(shape = [17,])
input2 = layers.Input(shape= [17,])
input3 = layers.Input(shape= [18,])
#input4 = layers.Input(shape= [17,])
encoder1 = layers.Dense(n_encoders, activation = "relu", kernel_regularizer = regularizers.l1(0.1))(input1)
encoder2 = layers.Dense(n_encoders, activation = "relu", kernel_regularizer = regularizers.l1(0.1))(input2)
encoder3 = layers.Dense(n_encoders, activation = "relu", kernel_regularizer = regularizers.l1(0.1))(input3)
#encoder4 = layers.Dense(n_encoders, activation = "relu", kernel_regularizer = regularizers.l1(0.1))(input4)
concat = layers.concatenate([encoder1,encoder2,encoder3])
encoder = layers.Dense(n_encoders, activation = "relu")(concat)
      
decoder = layers.Dense(17*2, activation = "sigmoid")(encoder)
autoencoder = Model(inputs = [input1,input2,input3], outputs = decoder)
autoencoder.compile(optimizer = "sgd", loss = "mse")#"categorical_crossentropy")
enco = Model(inputs = [input1,input2,input3], outputs = encoder)

inputs.append(norm_l2.fit_transform(I_locales).T)
autoencoder.fit(inputs,X,epochs = 50)

vl2 = np.c_[enco.predict(inputs),pipe.fit_transform(new_panel2[['vari0']])[:,-2:] ]
ae_km2 = copy(ae_km)
ae_km2.fit(vl2)
ae_km2.best_metrics_
mapa_grupos(ae_km2)
metrics3 = ind.compara_metricas(ae_km2, new_panel2)
grupos = []
for i in range(len(ae_km2.modelos)):
    g = ae_km2.modelos['modelo'].iloc[i].n_clusters
    grupos.append(g)
metrics3['grupos'] = grupos

metricas = pd.concat([metrics1,metrics2,metrics3])
metricas.corr()

metricas = metricas.reset_index(drop  = True)
metricas[['mi','ilq','hgr','sil']].plot()
metricas[['mi','hgr']].plot()

metricas.groupby('grupos').sum()[['mi','ilq','hgr','sil']].plot()
metricas[metricas.mi == metricas.mi.max()]
metricas.sort_values('mi').iloc[-10:]

metricas[metricas.hgr == metricas.hgr.max()]

ae_aglo = copy(aglo)
ae_aglo.fit(vl2)
ae_aglo.best_metrics_
mapa_grupos(ae_aglo)

### agrupan bien de acuerdo a los nuevos datos
np.random.seed = 646874
arm2 = area_region(covid_acum_geo.personas, regiones, peris = 10)
df2_nuevo = arm2.proceso_area_mult(2)
panel_df2_nuevo = arm2.convertir_a_panel(2)

indic = calcular_incicadores(df2, panel_df2)
indic_nuevos = calcular_incicadores(df2_nuevo, panel_df2_nuevo)

indic.Hg_relat(df2, km.best_model_['modelo'].iloc[0])
indic_nuevos.Hg_relat(df2, km.best_model_['modelo'].iloc[0])

indic.Hg_relat(df2, aglo.best_model_['modelo'].iloc[0])
indic_nuevos.Hg_relat(df2_nuevo, aglo.best_model_['modelo'].iloc[0])

indic.Hg_relat(df2, ae_km.best_model_['modelo'].iloc[0])
indic_nuevos.Hg_relat(df2_nuevo, ae_km.best_model_['modelo'].iloc[0])

indic.Hg_relat(df2, ae_km_moran.best_model_['modelo'].iloc[0])
indic_nuevos.Hg_relat(df2_nuevo, ae_km_moran.best_model_['modelo'].iloc[0])

indic.Hg_relat(df2, ae_aglo.best_model_['modelo'].iloc[0])
indic_nuevos.Hg_relat(df2_nuevo, ae_aglo.best_model_['modelo'].iloc[0])


###


input_train = []
input_val = []
for i in inputs:
    t,v  = sk.model_selection.train_test_split(i, random_state = 42)
    print(t.shape)
    print(i.shape)
    input_train.append(t)
    input_val.append(v)

X_train, X_val = sk.model_selection.train_test_split(X, random_state = 42)

autoencoder.fit(input_train, X_train, epochs = 50, validation_data = (input_val,X_val))
vl3 = np.c_[enco.predict(inputs),pipe.fit_transform(new_panel2[['vari0']])[:,-2:] ]

ae_km2.fit(vl3)
ae_km2.best_metrics_


pa2 = proceso_aleatorio(covid_acum_geo.personas,regiones, variables = 2).generar_datos(46546)
pa2.inputs_moran[2].shape

agru = agrupamientos(pa2,[km])
agru.clustering(autoencoder, enco)

n_encoders = 8
input1 = layers.Input(shape = [17,])
input2 = layers.Input(shape= [17,])
input3 = layers.Input(shape= [35,])
#input4 = layers.Input(shape= [17,])
encoder1 = layers.Dense(n_encoders, activation = "relu", kernel_regularizer = regularizers.l1(0.1))(input1)
encoder2 = layers.Dense(n_encoders, activation = "relu", kernel_regularizer = regularizers.l1(0.1))(input2)
encoder3 = layers.Dense(n_encoders, activation = "relu", kernel_regularizer = regularizers.l1(0.1))(input3)
#encoder4 = layers.Dense(n_encoders, activation = "relu", kernel_regularizer = regularizers.l1(0.1))(input4)
concat = layers.concatenate([encoder1,encoder2,encoder3])
encoder = layers.Dense(n_encoders, activation = "relu")(concat)
      
decoder = layers.Dense(17*2, activation = "sigmoid")(encoder)
autoencoder = Model(inputs = [input1,input2,input3], outputs = decoder)
autoencoder.compile(optimizer = "sgd", loss = "mse")#"categorical_crossentropy")
enco = Model(inputs = [input1,input2,input3], outputs = encoder)

#%%

inicio = time.time()
rdos = []
agrus = []
for i in np.random.randint(20,95645,2):
    pa = proceso_aleatorio(covid_acum_geo.personas, regiones, variables = 2).generar_datos(i)
    agru = agrupamientos(pa,[km,aglo])
    r = agru.clustering(autoencoder, enco)
    agrus.append(copy(agru))
    rdos.append(np.array(r))

fin = time.time()
np.array(rdos).mean(axis = 0)

mapa_grupos(agru.modelos[2])
import time 
ind = calcular_incicadores(df2,panel_df2, regiones)
inicio = time.time()
ind.compara_metricas(km, new_panel)
fin = time.time()
fin-inicio


inicio = time.time()
ind.Hg_relat(df, km.best_model_['modelo'].iloc[0])
#homog_relat2(df,regiones)
fin = time.time()

fin-inicio

inicio = time.time()
#ind.Hg_relat(df, km.best_model_['modelo'].iloc[0])
homog_relat2(df,km.best_model_['modelo'].iloc[0],'personas')
fin = time.time()

fin-inicio



incio = time.time()
silhouette_score(df.iloc[:,:-1], km.best_model_['modelo'].iloc[0].labels_)
fin = time.time()
fin-inicio
