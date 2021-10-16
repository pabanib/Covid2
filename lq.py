# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 16:07:59 2021

@author: pabanib

calcula el coeficiente de localización 

"""
import numpy as np
import pandas as pd
import scipy.stats as st
from copy import copy 
def lq(datos, campo, total):
    ind = datos[campo]/datos[total]
    indg = datos[campo].sum()/datos[total].sum()
    return [ind,indg,ind/indg]

def lq2(datos, campo, total):
    s = datos[campo]
    S = datos[campo].sum()
    theta = datos[total]/datos[total].sum()
    l = S*theta
    lq = s/l
    return s,l,lq

def intervalos(data, campo, total, a = .95):
    indices = lq(data, campo,total) 
    pi = indices[0]
    p = indices[1]
    ni = data[total]
    n = data[total].sum()
    
    var = (pi*(1-pi)/(ni*p**2))+((pi**2)*(1-p)/(n*p**3))+(2*(pi**2)*(1-pi)/(n*p**3))
    sd = np.sqrt(var)
    sd
    np.mean(indices[2])
    sdnorm = st.t.ppf((1+a)/2,2)*sd
    #sdnorm = st.norm.ppf((1+a)/2)*sd
    
    return pd.DataFrame([indices[2]-sdnorm,indices[2]+sdnorm.T,sdnorm,indices[2]]).T 

def intersec(a,b):
    
    if min(a) < max(b) and max(a)< min(b):
        return False
    elif min(b) < max(a) and max(b) < min(a):
        return False
    else:
        return True

def matrix_inters(x):
    #x debe ser un array de 2 dimensiones indicando los intervalos   
    l = len(x)
    dic = {}    
    for i in range(len(x)):
        v = []
        for j in range(len(x)):
            v.append(intersec(x[i],x[j]))
        dic[i]=v
    matrix = np.array(list(dic.values()))-np.eye(l)
    return matrix

def indice(x):
    return matrix_inters(x).sum()/(len(x)*(len(x)-1))

def matrix_inters_k(lista):
    # lista debe ser una lista de array de 2 dimensiones en donde cada array representa una variable con su intervalo
    variables = len(lista)
    l = len(lista[0])
    dic = {}
    for g in range(l):
        dic[g] = {}
        for i in range(variables):
            dic[g][i] = lista[i][g]
    
    x = {}
    for i in dic.keys():
        v = []
        for j in dic.keys():
            r = 1
            for k in dic[i].keys(): 
                if i == j:
                    r = r*1
                else:
                     r = r* intersec(dic[i][k],dic[j][k])
            v.append(r)
        x[i]=v
    matrix = np.array(list(x.values()))-np.eye(l)
    return matrix


def matrix_inters_k_debil(lista):
    # lista debe ser una lista de array de 2 dimensiones en donde cada array representa una variable con su intervalo
    variables = len(lista)
    l = len(lista[0])
    dic = {}
    for g in range(l):
        dic[g] = {}
        for i in range(variables):
            dic[g][i] = lista[i][g]
    
    x = {}
    for i in dic.keys():
        v = []
        for j in dic.keys():
            r = 0
            for k in dic[i].keys(): 
                if i == j:
                    r += 1
                else:
                     r += intersec(dic[i][k],dic[j][k])
            v.append(r/variables)
        x[i]=v
    matrix = np.array(list(x.values()))-np.eye(l)
    return matrix


class lq_peri():
    def __init__(self, X):
        # X debe ser un dataframe de tipo panel con N individuos y T periodos
        self.X = X
        self.peris = X.index.get_level_values(1).unique()
        
    def calc_interv_lq(self, grupos):
        # grupos debe ser un array de largo N 
        idx = pd.IndexSlice
        interv_lqs_ = []
        lqs_ = []
        #df = self.X.groupby([grupos, self.peris]).sum()
        for t in self.peris:
            df = self.X.loc[idx[:,t],:]
            df = df.groupby(grupos).sum()
            col = df.columns
            interv = intervalos(df,col[0],col[1])
            lqs_.append(interv[[3]].values)
            interv_lqs_.append(interv[[0,1]].values)
        self.lqs_ = lqs_
        self.interv_lqs = interv_lqs_
        return self.interv_lqs
    
    def calcular_indice(self,grupos):
        l = self.calc_interv_lq(grupos)
        matriz = matrix_inters_k(l)
        self.matriz_intersec = matriz
        return matriz.sum()/(len(matriz)*(len(matriz)-1))
    
    def calcular_indice_debil(self,grupos):
        l = self.calc_interv_lq(grupos)
        matriz = matrix_inters_k_debil(l)
        self.matriz_intersec_deb = matriz
        p = len(matriz)
        return (matriz.sum()/(p*(p-1)))+1/p
    
class region():
    
    def __init__(self, df_reg, poblacion, grupo = 'no'):
        
        if 'geometry' in df_reg.columns:
            self.geo = df_reg.geometry
            df_reg = df_reg.drop('geometry', axis = 1)
        if grupo == 'no':
            self.region = df_reg
        else:
            self.grupo = df_reg[grupo].unique
            self.region = df_reg.drop(grupo, axis = 1)
        self.poblacion = poblacion
        self.variables =  list(self.region.columns)       
        self.variables.remove(poblacion) 

    def calc_lq(self):
        lambdas = []
        lqs = []
        for i in self.variables:
            s,l,lq_ = lq2(self.region, i, self.poblacion)
            lambdas.append(l)
            lqs.append(lq_)
        lambdas = np.array(lambdas).T
        lqs = np.array(lqs).T
        self.s = s
        return lambdas, lqs

    def evaluar_var(self):
        self.lambdas, self.lqs = self.calc_lq()
        reg = self.region[self.variables].values
        ssd = ((reg-self.lambdas )**2).mean(axis= 0)
        tssd = ssd.mean()
        return ssd,tssd
        
    def evaluar_prom(self):
        self.lambdas, self.lqs = self.calc_lq()
        reg = self.region[self.variables].values
        ssd = (reg.mean(axis = 1)-self.lambdas.mean(axis = 1))**2
        tssd = ssd.mean()
        return ssd, tssd
    
class particion():
    def __init__(self, grupos):
        self.grupos = grupos
        
    def fit(self,X):
        return self
    def transform(self, X):
        lista = X[self.grupos].unique()
        regiones = {}
        for i in lista:
            r = X[X[self.grupos]==i]
            regiones[i] = r
        return regiones
    
def homogeneidad_particion(df, part, pobl= 'personas'):
    dic = part.transform(df)
    grupo = part.grupos
    res = []
    for k in dic.keys():
        r = region(dic[k],pobl,grupo)
        rr = r.evaluar_prom()[1]
        res.append(rr)
    return  np.mean(np.array(res))

def ss(df):
    medias = df.mean(axis = 0)
    ssd = []
    for i in range(df.shape[1]):
        var = df[:,i]-medias[i]
        var2 = var**2
        ssd.append(sum(var2))
    ssd = np.array(ssd)
    return  sum(ssd)

def SSD(DF, columna, atrib):
    df = DF[atrib]
    from sklearn import preprocessing
    escalado = preprocessing.StandardScaler().fit(df)
    TSS = ss(escalado.transform(df))
    #TSS = ss(df)
    WSS = []
    for i in DF[columna].unique():
        d = df[DF[columna] == i]
        d = escalado.transform(d)
        s = ss(d)
        WSS.append(s)
    WSS = np.sum(np.array(WSS))
    BSS = TSS-WSS
    RBTSS = BSS/TSS
    return {'TSS': TSS, 'WSS': WSS, 'BSS': BSS, 'RBTSS':RBTSS}


class evaluaciones_lq():
    
    def __init__(self, grupos, variables, poblacion):
        from procesos import peri_columna
             
        self.grupo = grupos
        self.variables = variables
        self.poblacion = poblacion
        self.peri_col = peri_columna()
        
    def ajustar_datos(self, X):
        # X debe ser un DF tipo panel de (NxT)x K 
        # además de la variable de población
        per = X.index.get_level_values(1).unique()[-1]
        idx = pd.IndexSlice
        var_pobl = X.loc[idx[:,per],self.poblacion]
        self.bd_globales = {}
        for i in self.variables:
            bd = X[[i, self.poblacion]]
            self.bd_globales[i] = bd
            
        self.bd_locales = {}
        for i in self.variables:
            bd = self.peri_col.fit_transform(X[[i]])
            bd = pd.DataFrame(bd)
            bd[self.poblacion] = var_pobl.values
            self.bd_locales[i] = bd
        
        
    def lq_globales(self, grupo):
        
        rdos_glob = {}
        datos = self.bd_globales
        for k in self.variables:
            var = {}
            lqperi = lq_peri(datos[k])
            ind = [lqperi.calcular_indice(grupo),lqperi.calcular_indice_debil(grupo)]
            var['indices'] = ind
            var['lqperi'] = copy(lqperi)
            rdos_glob[k] = var
            
        self.rdos_glob = rdos_glob
        
    def particion(self,X,grupo):
        
        lista = np.unique(grupo)
        regiones = {}
        for i in lista:
            r = X[grupo == i]
            regiones[i] = r
        return regiones
    
    def lq_locales(self, grupo):
        self.rdos_loc = {}
        for i in self.variables:
            var = {}
            bd = self.bd_locales[i]
            regiones = self.particion(bd,grupo)
            res =[]
            region_ = {}
            for k in regiones.keys():
                reg ={}
                r = region(regiones[k],self.poblacion)
                rr = r.evaluar_prom()[1]
                ind = [r.evaluar_prom()[1], r.evaluar_var()[1]]
                res.append(rr)
                reg['indices'] = ind # Diferencia entre el prom lambda y el lambda teorico
                reg['region'] = copy(r)
                region_[k] = reg
            var['regiones'] = region_
            var['homogeneidad']= np.mean(np.array(res))
            self.rdos_loc[i] = var
    def calcular_indices(self, X):
        #devuelve el índice débil y la homogeneidad de la partición
        self.ajustar_datos(X)
        self.lq_globales(self.grupo)
        self.lq_locales(self.grupo)
        rdos = {}
        for v in self.variables:
            ind_debil = self.rdos_glob[v]['indices'][1]
            homog = self.rdos_loc[v]['homogeneidad']
            rdos[v] = ind_debil,np.sqrt(homog)
        return rdos
         
class evaluaciones_grupos():
        
    def evaluar(self,X,grupos):
        # Grupos tiene que ser un array n filas (cantidad de casos) y k columnas (metodos)
        rdos_grupos = []
        obj_grupos = {}
        self.ajustar_datos(X)
        for j in range(grupos.shape[1]):
           col = grupos[:,j]       
           self.lq_globales(col)
           self.lq_locales(col)
           
        
#%% 

# import geopandas as gpd
# import os

# dir_principal = os.getcwd()
# dir_datos = dir_principal+'\\datos'

# covid = gpd.read_file(dir_datos+'/covid_periodos.shp', index = True)
# covid = covid.set_index(['link','mes']).sort_index(level = 0)
# covid = covid.loc[pd.IndexSlice[:,'2020-03':],:]
# covid = covid.to_crs('POSGAR94')

# # Separamos los campos geometricos del dataframe
# geo = covid.loc[pd.IndexSlice[:,'2021-01'],'geometry']
# geo = geo.reset_index(level = 'mes', drop = True)
# centroides = covid.loc[pd.IndexSlice[:,'2021-01'],'geometry'].to_crs('POSGAR94').centroid
# centroides = centroides.reset_index(level = 'mes', drop = True)
# print("las cordenadas CRS son: "+str(geo.crs))
# codiprov = covid.loc[pd.IndexSlice[:,'2021-01'],['codpcia','departamen','provincia']]


# columnas = ['clasificac', 'fallecido']

# # Variables acumuladas a partir del mes que todas tienen al menos 1 

# covid_acum = covid[columnas].groupby(covid.index.get_level_values(0)).cumsum()
# # buscamos el mes en que todos los dptos tienen al menos 1 contagio
# mes = 0
# valor = True
# while valor == True:
#     Mes = covid.index.get_level_values(1).unique()[mes]
#     valor = np.any(covid_acum.loc[pd.IndexSlice[:,Mes],'clasificac'] == 0)
#     mes +=1
# print("El mes desde el cuál todos los dptos tienen al menos 1 contagiado es: "+str(Mes))
# covid_acum['personas'] = covid.personas

# covid2 = covid_acum.loc[pd.IndexSlice[:,Mes:],:]
# covid_ult_mes = covid_acum.loc[pd.IndexSlice[:,'2021-07'],:]
# covid_ult_mes = covid_ult_mes.reset_index(level = 'mes', drop = True)

# #casos cada 10 mil habitantes
# fallecidos = covid2.fallecido/(covid.loc[pd.IndexSlice[:,Mes:],:].personas/10000)
# positivos = covid2.clasificac/(covid.loc[pd.IndexSlice[:,Mes:],:].personas/10000)
# falle = covid2.fallecido/(covid2.personas/10000)

# # Calculamos el coeficiente de localización
# from lq import *
# lq_ = lq(covid2,'fallecido','clasificac')
# lq_fall_conf = lq_[2]
# ind_fall_conf = lq_[0]

# #la variable se elige para comparar con diferentes opciones
# variable = fallecidos #covid2[['clasificac','personas']]

#%%

# eva = evaluaciones_lq(np.random.randint(0,8,525), ['fallecido','clasificac'], 'personas')
# eva.ajustar_datos(covid_acum)        
# eva.lq_globales(eva.grupo)        
# eva.rdos_glob['fallecido']    
# #eva.rdos_['clasificac']

# eva.lq_locales(eva.grupo)
# eva.rdos_loc['fallecido']['homogeneidad']
# eva.rdos_loc['fallecido']['regiones'].keys()
# eva.rdos_loc['fallecido']['regiones'][0]['indices']
# eva.rdos_loc['fallecido']['regiones'][1]['indices']

# eva.calcular_indices(covid_acum)
