# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 19:11:10 2021
@author: pabanib

Evalúa varios metodos de clustering bajo alguna metrica probando varios hiperparámetros


"""
#from sklearn.cluster import KMeans, AgglomerativeClustering
from copy import copy
import pandas as pd
import time
import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
import procesos
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

class metodo():
    def __init__(self, metodo, param,metric):
        
        # Parameters: 
        # metodo: It is the clustering metodology of scikit learn
        # param: Dictionary of  metodos's parameters 
        # metrci: Dictionary of {name: callable()}
        self.metodo = metodo
        self.param = param
        self.metric = metric
    def grid(self):
        from sklearn.model_selection import ParameterGrid
        parametros = list(ParameterGrid(self.param))
        return parametros
    
    def modelo(self, diccionario_parametros):
        dic = diccionario_parametros
        kmean = {'n_clusters': 8, 'init': 'k-means++', 'n_init': 10, 'max_iter':300, 'tol': 0.0001}
        aglo = {'n_clusters': 8, 'affinity': 'euclidean', 'connectivity':None, 'linkage': 'ward'}
        tskmean = {'n_clusters': 8, 'init': 'k-means++', 'n_init': 1, 'max_iter':300, 'tol': 0.0001, 'metric': "dtw"}
        kshape = {'n_clusters': 8,  'max_iter':100, 'tol': 0.0001, 'init': 'random'}
        if str(type(self.metodo)) == "<class 'sklearn.cluster._kmeans.KMeans'>":
            for k in kmean.keys():
                try:
                    kmean[k] = dic[k] 
                except KeyError:
                    pass                  
            metod = copy(self.metodo)    
            metod.__init__(n_clusters = kmean['n_clusters'], init = kmean['init'], n_init = kmean['n_init'], tol = kmean['tol'])
            return metod
        
        elif str(type(self.metodo)) == "<class 'sklearn.cluster._agglomerative.AgglomerativeClustering'>":
            for k in aglo.keys():
                try:
                    aglo[k] = dic[k] 
                except KeyError:
                    pass #print('la clave {} no se encuentra'.format(k))
            metod = copy(self.metodo)    
            metod.__init__(n_clusters = aglo['n_clusters'], affinity = aglo['affinity'], connectivity = aglo['connectivity'],linkage = aglo['linkage'] )
            return metod
        
        elif str(type(self.metodo)) == "<class 'tslearn.clustering.kmeans.TimeSeriesKMeans'>":
            for k in tskmean.keys():
                try:
                    tskmean[k] = dic[k] 
                except KeyError:
                    pass                  
            metod = copy(self.metodo)    
            metod.__init__(n_clusters = tskmean['n_clusters'], init = tskmean['init'], n_init = tskmean['n_init'], tol = tskmean['tol'], metric = "dtw")
            return metod
        
        elif str(type(self.metodo)) == "<class 'tslearn.clustering.kshape.KShape'>":
            for k in kshape.keys():
                try:
                    kshape[k] = dic[k] 
                except KeyError:
                    pass                  
            metod = copy(self.metodo)    
            metod.__init__(n_clusters = kshape['n_clusters'], init = kshape['init'], tol = tskmean['tol'] )
            return metod
        
        else:
            print('no se reconoce el método')
            
            
    def fit(self, data):
        p = self.grid()
        self.parametros = p
        modelos = []
        metrics = []
        for dic in p:
            try:
                model = self.modelo(dic)
                inicio = time.time()
                model.fit(data)
                rdo = self.calc_metric(data, model)
                tiempo = time.time()-inicio
                modelos.append([model,tiempo, sys.getsizeof(model)])
                metrics.append(rdo.values[0])
            except:
                print('fallo el sig dic:')
                print(dic)
        modelos = pd.DataFrame(modelos, columns = ('modelo', 'tiempo','Tamaño'))
        self.metrics = pd.DataFrame(metrics, columns = rdo.columns)
        self.modelos = modelos
        self.best_model_ = self.modelos.iloc[self.best_model(self.metrics).index]
        self.best_metrics_ = self.metrics.iloc[self.best_model(self.metrics).index]
        self.best_time_ = self.modelos[self.modelos.tiempo == self.modelos.tiempo.min()]
        
    def calc_metric(self,data,model):
        #every metric maid have parameters: (data, model)
        metric_result = []
        for k in self.metric.keys():
            metric_result.append(self.metric[k](data,model))
        
        self.metric_result=pd.DataFrame(metric_result, index = self.metric.keys())
        #self.metric_result = pd.DataFrame(metric_result)
        return self.metric_result.T
    
    def best_model(self, metricas):
        # the best model have the most larger sum of normalizer metrics
        from sklearn.preprocessing import StandardScaler
        st = StandardScaler()
        m= st.fit_transform(metricas)
        return metricas[m.sum(axis = 1) == m.sum(axis = 1).max()]
        
class clustering_autoencoder():

    def __init__(self,centroides, n_encoders = 5):
        self.n_encoders = n_encoders
        self.centroides = centroides
        self.pipe_coord = Pipeline([
            ('periodos', procesos.peri_columna()),
            ('coordenadas', procesos.agrega_centroides(centroides))
                ]              
            )
                       
        self.norm_l1 = Pipeline([
             ('std_scale', StandardScaler()),
             ('norm_l1', Normalizer('l1'))])
        self.norm_l2 = Pipeline([
             ('std_scale', StandardScaler()),
            ('norm_l2', Normalizer('l2'))])
        
        self.pipe = Pipeline([
            ('coord', self.pipe_coord),
            ('norml1', self.norm_l1)])
  
        
    def ajustar_datos(self, x, n_variables = 1):
        X = self.pipe.fit_transform(x)
        geo = X[:,-2:]
        dat = X[:,:-2]
        
        if n_variables == 1:
            X = dat
        else:
            n = dat.shape[0]
            k = dat.shape[1]
            X = dat.reshape(n,n_variables,int(round(k/n_variables,0)))
            
        return X, geo

    def model1(self, shape_input):
        
        entrada = layers.Input(shape = (shape_input,))
        encoder = layers.Dense(self.n_encoders, activation = "relu", kernel_regularizer = regularizers.l1(0.1))(entrada)
        #encoder = layers.Dense(5, activation = "relu")(encoder)
        decoder = layers.Dense(shape_input, activation = "sigmoid")(encoder)
        
        self.autoencoder = Model(inputs = entrada, outputs = decoder)
        self.autoencoder.compile(optimizer = "sgd", loss = "categorical_crossentropy")
        self.enco = Model(inputs = entrada, outputs = encoder)
        
    def model2(self, shape_input):
        #shape input es un array o lista con los tamaños para las entradas
        
        input1 = layers.Input(shape = (shape_input[0],))
        input2 = layers.Input(shape = (shape_input[1],))
        encoder1 = layers.Dense(self.n_encoders, activation = "relu", kernel_regularizer = regularizers.l1(0.1))(input1)
        encoder2 = layers.Dense(self.n_encoders, activation = "relu", kernel_regularizer = regularizers.l1(0.1))(input2)
        concat = layers.concatenate([encoder1,encoder2])
        encoder = layers.Dense(self.n_encoders, activation = "relu")(concat)
        
        
        decoder = layers.Dense(sum(shape_input), activation = "sigmoid")(encoder)
        self.autoencoder = Model(inputs = [input1,input2], outputs = decoder)
        self.autoencoder.compile(optimizer = "sgd", loss = "categorical_crossentropy")
        self.enco = Model(inputs = [input1,input2], outputs = encoder)

    def fit_autoencoder(self,x, n_variables):
        X,geo = self.ajustar_datos(x,n_variables)
        if n_variables == 1:
            modelo = self.model1(X.shape[1])
            self.autoencoder.fit(X,X, epochs = 50)
            encoded_valores = self.enco.predict(X)

        else:
            modelo = self.model2([X.shape[2],X.shape[2]])
            X1 = X[:,0,:]
            X2 = X[:,1,:]
            X = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
            self.autoencoder.fit((X1,X2),X, epochs = 50)
            encoded_valores = self.enco.predict((X1,X2))
        
        import numpy as np
        self.encoded_valores = encoded_valores
        self.geo = geo 
                
        return np.c_[encoded_valores,geo]



















