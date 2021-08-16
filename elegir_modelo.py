# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 19:11:10 2021
@author: pabanib

Evalúa varios metodos de clustering bajo alguna metrica probando varios hiperparámetros


"""
from sklearn.cluster import KMeans, AgglomerativeClustering
from copy import copy
import pandas as pd
import time
import sys
class metodo():
    def __init__(self, metodo, param,metric):
        
        # Parameters: 
        # metodo: It is the clustering metodology of scikit learn
        # param: Dictionary of  metodos's parameters 
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
                    print('la clave {} no se encuentra'.format(k))
            metod = copy(self.metodo)    
            metod.__init__(n_clusters = aglo['n_clusters'], affinity = aglo['affinity'], connectivity = aglo['connectivity'],linkage = aglo['linkage'] )
            return metod
        
        else:
            print('no se reconoce el método')
            
            
    def fit(self, data):
        p = self.grid()
        self.parametros = p
        modelos = []
        for dic in p:
            try:
                model = self.modelo(dic)
                inicio = time.time()
                model.fit(data)
                rdo = self.metric(data, model)
                tiempo = time.time()-inicio
                modelos.append([model,rdo,tiempo, sys.getsizeof(model)])
            except:
                print('fallo el sig dic:')
                print(dic)
        modelos = pd.DataFrame(modelos, columns = ('modelo','Metrica', 'tiempo','Tamaño'))
        self.modelos = modelos
        self.best_model_ = self.modelos[self.modelos.Metrica == self.modelos.Metrica.max()]
        self.best_time_ = self.modelos[self.modelos.tiempo == self.modelos.tiempo.min()]
    
