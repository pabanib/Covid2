# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 10:20:32 2022

@author: Pablo
"""
import numpy as np
import pandas as pd
import geopandas as gpd 
import lq
import elegir_modelo as em
import sklearn
import procesos
from libpysal.weights import Queen, Rook, KNN
from esda.moran import Moran_Local
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import copy

class Datos():
    
    def __init__(self, df, variables, poblacion):
        
        assert isinstance(df,gpd.geodataframe.GeoDataFrame), "El DataFrame no es un GeoDataFrame"
        
        if isinstance(df.index,pd.core.indexes.multi.MultiIndex):
            self.panel_df = df
            p = df.index.get_level_values(1).unique()[0]
            self.geo = df.loc[pd.IndexSlice[:,p],'geometry']
            self.df = False
            
        elif  isinstance(df.index,pd.core.indexes.base.Index):
            self.df = df
            self.geo = df[['geometry']]
            self.panel_df = [False]
        
                
        
        self.variables = variables
        self.poblacion = poblacion
        self.centroides = self.geo.centroid.reset_index(level = 1, drop = True)
        self.add_centroides = procesos.agrega_centroides(self.centroides)
        self.coord_centroides = self.add_centroides.coordenadas(self.centroides)
        
    def add_pipeline(self, pipeline):
        
        assert isinstance( pipeline,sklearn.pipeline.Pipeline ), 'Debe ingresar un pipeline de sklearn'
        self.pipeline = pipeline
        
    def ajustar_datos(self):

        pass
       
    def agregar_metrica(self,nombre, metrica):
        
        self.metricas[nombre] = metrica

    def convertir_a_panel(self, peris,variables = 1):
        
        df = self.df.drop('geometry', axis = 1)
        if variables > 1:
            df_ = df[df.columns[:-1]].stack()
            if isinstance(df.columns, pd.core.indexes.multi.MultiIndex):
                columnas = list(df_.columns.get_level_values(0).unique())
            else:
                columnas = []
                for i in range(variables):
                    columnas.append('vari'+str(i))
                              
        else:
            df_ = df[df.columns[:list(df.columns).index(self.poblacion)]].stack()
            columnas = list(['vari1'])
        columnas.append(self.poblacion)
        p = np.array([[df[self.poblacion].values,]]*peris)
        
        X = np.c_[df_.values, p.T.reshape(-1,1)]
        return pd.DataFrame(X, index = df_.index, columns = columnas)        
    
    def convertir_a_df(self, panel_df,variables):
        
        df = panel_df[variables].unstack()
        p = panel_df.index.get_level_values(1).unique()[0]
        df[self.poblacion] = panel_df.loc[pd.IndexSlice[:,p],:].set_index(df.index)[self.poblacion]
        
        return df
       
    def agregar_geometria(self):
        df = self.convertir_a_df(self.panel_df, self.variables)
        geo = self.geo.copy()
        geo.index = df.index
        df = gpd.GeoDataFrame(df, geometry = geo)
        
        return df
       
    def matriz_W(self, k = 6):
                
        df = self.agregar_geometria()
        self.W_queen = Queen.from_dataframe(df)
        self.W_rook = Rook.from_dataframe(df)
        self.W_knn = KNN.from_dataframe(df, k = k)
        
    def calc_Imoran(self, W):
        if self.df == False:
            df = self.convertir_a_df(self.panel_df, self.variables)
        else:
            df = self.df.drop('geometry')
        
        I_locales = []
        for c in df.columns:
            I = Moran_Local(df[c], W).Is
            I_locales.append(I)
        
        return np.array(I_locales).T
    def calc_prom_vec(self, W):
        if self.df == False:
            df = self.convertir_a_df(self.panel_df, self.variables)
        else:
            df = self.df.drop('geometry')
        
        prom_vec = []
        for c in df.columns:
            p =  W.sparse.toarray()@ df[c]
            prom_vec.append(p)
        
        return np.array(prom_vec).T
    def separar_variables(self):
        if self.df == False:
            df = self.convertir_a_df(self.panel_df, self.variables)
        else:
            df = self.df.drop('geometry')
            
        dfs = {}
        for i in self.variables:
           d = df.loc[:,pd.IndexSlice[i,:]] 
           dfs[i] = d
           
        dfs[self.poblacion] =  df.loc[:,pd.IndexSlice[self.poblacion,:]] 
        return dfs

class dic_datos():
    def __init__(self, pipeline):
        assert isinstance( pipeline,sklearn.pipeline.Pipeline ), 'Debe ingresar un pipeline de sklearn'
        self.pipeline = pipeline
        self.dic = {}
            
    def aggregar_data(self, key, df):
        
        self.dic[key] = self.pipeline.fit_transform(df)
    
    def retornar_dfs(self, todo = False, separado = []):
        
        dfs = []
        if todo:
            l = list(self.dic.keys())
            df = self.dic[l[0]]
            for k in l[1:]:
                df = np.c_[df, self.dic[k]]
            dfs.append(df)    
        else:
            for i in separado:
                if isinstance(i, list):
                    df = self.dic[i[0]]
                    for j in i[1:]:
                        df = np.c_[df,self.dic[j]]
                    dfs.append(df)
                else:
                    df = self.dic[i]
                    dfs.append(df)
                    
        return dfs
    
class autoencoders():
    
    def __init__(self, n_encoders):
        self.n_encoders = n_encoders
    
    def crear_autoencoders(self, lista_df):
        n = len(lista_df)
        shape_inp = []
        self.inputs = []
        self.encoders = []
        for i in range(n):
            s = lista_df[i].shape[1]
            shape_inp.append(s)
            inp = layers.Input(shape = [s,])
            enc = layers.Dense(self.n_encoders, activation = "relu", kernel_regularizer = regularizers.l1(0.1))(inp)        
            self.inputs.append(inp)
            self.encoders.append(enc)
            
        concat = layers.concatenate(self.encoders)
        encoder = layers.Dense(self.n_encoders, activation = "relu" )(concat)
        
        decoder = layers.Dense(sum(shape_inp), activation = "sigmoid")(encoder)
        
        self.autoencoder = Model(inputs = self.inputs, outputs = decoder)
        self.autoencoder.compile(optimizer = "sgd", loss = "mse")
        
        self.enco = Model(inputs = self.inputs, outputs = encoder)
        
        return self
    
    def fit_autoencoders(self, inputs, outputs, epochs):
        
        self.crear_autoencoders(inputs)
        self.autoencoder.fit(inputs,outputs,epochs = epochs)
        return self.enco.predict(inputs)

class calcular_metricas():
    def __init__(self, datos, regiones = []):
        
        #assert isinstance(datos, Datos), "Datos debe ser un objeto tipo regionalizacion.datos"
        self.datos = datos
        if all(datos.panel_df) == False:
            df = datos.df
            print("convertir a panel los datos")
        
        else:
            df = datos.convertir_a_df(datos.panel_df, datos.variables)
            self.lqperis = []
            for c in datos.variables:
                lq_ = lq.lq_peri(datos.panel_df[[c,datos.poblacion]])
                self.lqperis.append(copy.copy(lq_))
                   
        
        self.df = df
        self.pobl = datos.poblacion
        self.regiones = regiones
    
    def obtener_panel(self, periodos, variables):
        panel_df = self.datos.convertir_a_panel(periodos, variables)
        self.lqperis = []
        for c in self.datos.variables:
            lq_ = lq.lq_peri(panel_df[[c,self.datos.poblacion]])
            self.lqperis.append(copy(lq_))
                                
    
    def indice_lq(self, X, grupos):
        
        ind = 0
        for i in self.lqperis:
            ind += i.calcular_indice_debil(grupos)
       
        return 1-((ind)/len(self.lqperis))
   
    
    def Hg_relat(self,X,grupos):
    
        return 1-lq.homog_relat2(self.df,grupos, list(self.df.columns)[-1])

    def MI(self,X,grupos):
        real = self.regiones
        if len(real) == 0:
            m = np.nan
        else:
            m = sklearn.metrics.adjusted_mutual_info_score(real,grupos) 
        return m

    def compara_metricas(self,modelo, datos):
        data = datos
        result = []
        for m in range(len(modelo.modelos)):
            model = modelo.modelos.iloc[m]['modelo']
            mi = self.MI(data,model.labels_)
            ilq = self.indice_lq(data, model.labels_)
            hgr = self.Hg_relat(data, model.labels_)
            sil = siluetas(data, model.labels_)
            cal = calinski(data, model.labels_)
            dav = inv_davies_bouldin(data,model.labels_)
            try:
                iner = model.inertia_
            except:
                iner = np.nan
            result.append(np.array([mi,ilq,hgr,sil,cal,dav,iner]))
        result = pd.DataFrame(result, columns = ['mi','ilq','hgr','sil','cal','dav'])
        return result
    def calc_metricas(self,datos, grupos):
        data = datos
        result = []
        
        mi = self.MI(data,grupos)
        ilq = self.indice_lq(data, grupos)
        hgr = self.Hg_relat(data, grupos)
        sil = siluetas(data, grupos)
        cal = calinski(data, grupos)
        dav = inv_davies_bouldin(data,grupos)
        result.append(np.array([mi,ilq,hgr,sil,cal,dav]))
        result = pd.DataFrame(result, columns = ['mi','ilq','hgr','sil','cal','dav'])
        return result


def siluetas(X, grupos):
    return sklearn.metrics.silhouette_score(X, grupos)
def calinski(x, grupos):
    return sklearn.metrics.calinski_harabasz_score(x, grupos)
def inv_davies_bouldin(X,grupos):
    return (sklearn.metrics.davies_bouldin_score(X,grupos))**-1
            
        
#%%    
    
dat = Datos(covid, ['fallecido','clasificac'], 'personas')
dat.panel_df
dat.convertir_a_df(dat.panel_df, dat.variables).loc[:,pd.IndexSlice['fallecido',:]]
dat.matriz_W()
dat.calc_Imoran(dat.W_knn)
dat.calc_prom_vec(dat.W_knn).shape

dat.separar_variables()
dat.coord_centroides


norm_l2 = sklearn.pipeline.Pipeline([
     ('std_scale', sklearn.preprocessing.StandardScaler()),
    ('norm_l2', sklearn.preprocessing.Normalizer('l2'))])

dic = dic_datos(norm_l2)

dic.aggregar_data('datos', dat.convertir_a_df(dat.panel_df, dat.variables))
dic.aggregar_data('centroides', dat.calc_Imoran(dat.W_knn))
dic.aggregar_data('geo2', dat.calc_prom_vec(dat.W_knn))

dic.retornar_dfs(todo = True)[0].shape

dic.retornar_dfs(separado = [['datos'],['centroides','geo2']])[1].shape
d = dic.retornar_dfs(separado = [['datos'],['centroides','geo2']])

auto = autoencoders(2)
auto.crear_autoencoders(d)

auto.autoencoder.fit( d, dic.retornar_dfs(todo = True), epochs = 50)
auto.autoencoder.fit( d, dic.retornar_dfs('datos'), epochs = 50)

auto.enco.predict(d).mean(axis = 0)


metricas = calcular_metricas(dat)
metricas.calc_metricas(dic.dic['datos'], codiprov.codpcia.values)
list(metricas.df.columns)[-1]

dic.dic['datos']

dat.geo.index = dat.geo.index.get_level_values(0)
dat.geo

geo = gpd.GeoDataFrame(dat.geo)
Queen.from_dataframe(geo)
Rook.from_dataframe(geo)
df = dat.convertir_a_df(dat.panel_df, dat.variables)
df = gpd.GeoDataFrame(df,geometry = dat.geo)
KNN.from_dataframe(df,k = 6)
df
dat.geo

import procesos
p = procesos.peri_columna()

c = covid[['fallecido','clasificac']].unstack()
ind = covid.index.get_level_values(0)
c['personas'] = covid.loc[pd.IndexSlice[:,'2020-03'],:].set_index(c.index)['personas']


c = gpd.GeoDataFrame(c, geometry  = geo)

dat2 = datos(c, ['fallecido','clasificac'],'personas')
dat2.df.drop('geometry', axis = 1)
dat2.panel_df
dat2.convertir_a_panel(17,3)
