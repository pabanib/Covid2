# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 16:07:59 2021

@author: pabanib

calcula el coeficiente de localización 

"""
import numpy as np
import pandas as pd
import scipy.stats as st
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
        lqs_ = []
        #df = self.X.groupby([grupos, self.peris]).sum()
        for t in self.peris:
            df = self.X.loc[idx[:,t],:]
            df = df.groupby(grupos).sum()
            col = df.columns
            interv = intervalos(df,col[0],col[1])[[0,1]].fillna(1)
            lqs_.append(interv.values)
        self.interv_lqs = lqs_
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


