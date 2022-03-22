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

var = cov.reg.map({1:0.0001,2:0.001,3:0.1, 4:0.3, 5:0.7})* cov.personas


from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import regionalizacion as reg


var = pd.concat([var,cov.personas], axis = 1)
var = var.rename(columns = {0:'var1'})
var = var.droplevel('mes')
var = gpd.GeoDataFrame(var, geometry = geo)

#creamos el objeto Datos, este nos genera la mayoría de los datos necesarios
sim1 = reg.Datos(var, ['var1'],'personas')

sim1.matriz_W(6)
sim1.calc_Imoran(sim1.W_queen) == sim1.calc_Imoran(sim1.W_queen, var[['var1','personas']].values)

sim1.calc_prom_vec(sim1.W_queen)
sim1.calc_prom_vec(sim1.W_queen, var[['var1','personas']].values)

#creamos un obbjeto dic_datos que sirve para normalizar datos y seleccionar distintas bases
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer

pipe = Pipeline([('standard', StandardScaler()),
                 ('normalizer', Normalizer('l2'))
                 ])

class entorno(reg.Datos, reg.dic_datos, reg.metodo):
    
    def __init__(self, df, variables, poblacion, pipeline):
        reg.Datos.__init__(self,df, variables,poblacion)
        reg.dic_datos.__init__(self,pipeline)
        
        dat = reg.Datos(df, variables, poblacion)
        self.metric = reg.calcular_metricas(dat)

    
        
ent1 = entorno(var, ['var1'],'personas', pipe)
ent1.metric.obtener_panel(1, 1)
ent1.df

ent2 = entorno(covid,['clasificac'],'personas',pipe)
ent2.panel_df
ent2.convertir_a_df(ent2.panel_df, ent2.variables)

dic = reg.dic_datos(pipe)

dic.aggregar_data('datos', sim1.df.drop('geometry', axis = 1))
dic.dic['datos'].std(axis = 0)

dic.aggregar_data('coord', sim1.coord_centroides)

metrics = reg.calcular_metricas(sim1)
metrics.obtener_panel(1,1)

param = {'n_clusters' : [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}
metricas = {'hg': metrics.Hg_relat, 'lq': metrics.indice_lq }

metrics.Hg_relat(dic.retornar_dfs(True), np.random.randint(0,5,525))

km = reg.metodo(KMeans(),param,metricas)

km.fit(dic.retornar_dfs(todo = True)[0])

km.best_metrics_
km.best_model_
km.metrics

reg.mapa_grupos(cov,km)

sim1.matriz_W(6)
dic.aggregar_data('I-moran', sim1.calc_Imoran(sim1.W_knn))

dic.retornar_dfs(True)[0].shape
km.fit(dic.retornar_dfs(True)[0])

reg.mapa_grupos(var, km)

aglo = reg.metodo(AgglomerativeClustering(), param, metricas)

aglo.fit(dic.retornar_dfs(True)[0])
reg.mapa_grupos(cov,aglo)
aglo.best_metrics_

ae = reg.autoencoders(1)
enco = ae.fit_autoencoders(dic.retornar_dfs(separado = ['datos', 'coord','I-moran']), dic.retornar_dfs(True), epochs=50)
enco
km.fit(enco)
reg.mapa_grupos(var, km)

aglo.fit(enco)
reg.mapa_grupos(var, aglo)

dic.aggregar_data('prom_v', sim1.calc_prom_vec(sim1.W_knn))

enco = ae.fit_autoencoders(dic.retornar_dfs(separado = ['datos', 'coord','I-moran','prom_v']), dic.retornar_dfs(True), epochs=50)
km.fit(enco)
reg.mapa_grupos(var, km)
aglo.fit(enco)
reg.mapa_grupos(var, aglo)

dic.retornar_dfs(todo = False,separado = ['coord'])[0].shape

dic.aggregar_data('porcent', (sim1.df['var1']/sim1.df['personas']).values.reshape(-1,1), ajustar = False)

km.fit(dic.retornar_dfs(separado = [['datos','porcent','coord']])[0])
reg.mapa_grupos(var, km)

aglo.fit(dic.retornar_dfs(separado = [['datos','porcent','coord']])[0])
reg.mapa_grupos(var, aglo)

km.fit(dic.retornar_dfs(separado = [['datos','porcent','coord','I-moran']])[0])
reg.mapa_grupos(var, km)

aglo.fit(dic.retornar_dfs(separado = [['datos','porcent','coord','I-moran']])[0])
reg.mapa_grupos(var, aglo)

km.fit(dic.retornar_dfs(separado = [['datos','porcent','coord','I-moran','prom_v']])[0])
reg.mapa_grupos(var, km)

aglo.fit(dic.retornar_dfs(separado = [['datos','porcent','coord','I-moran','prom_v']])[0])
reg.mapa_grupos(var, aglo)

ae = reg.autoencoders(1)
enco = ae.fit_autoencoders(dic.retornar_dfs(separado = ['datos','porcent','coord','I-moran','prom_v']), dic.retornar_dfs(separado = [['datos','porcent','coord','I-moran','prom_v']]), epochs = 50)

km.fit(enco)
reg.mapa_grupos(var, km)
aglo.fit(enco)
reg.mapa_grupos(var, aglo)

ae = reg.autoencoders(2)
enco = ae.fit_autoencoders(dic.retornar_dfs(separado = [['datos','porcent'],['I-moran','prom_v'],'coord']), dic.retornar_dfs(separado = [['datos','porcent','I-moran','prom_v','coord']]), epochs = 50)
#enco = np.c_[enco, dic.dic['coord']]

km.fit(enco)
reg.mapa_grupos(var, km)
aglo.fit(enco)
reg.mapa_grupos(var, aglo)

km.fit(dic.retornar_dfs(separado = ['datos','porcent','coord'])[0])
reg.mapa_grupos(var, km)
km.best_metrics_

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(2)
dic.aggregar_data('poly',poly.fit_transform(dic.dic['datos']))

km.fit(dic.retornar_dfs(separado = ['poly','coord','I-moran','prom_v'])[0])
reg.mapa_grupos(cov, km)
km.best_metrics_

enco = ae.fit_autoencoders(dic.retornar_dfs(separado = [['poly'],['I-moran','prom_v']]), dic.retornar_dfs(separado = [['poly','I-moran','prom_v']]), epochs = 50)
enco = np.c_[enco,dic.dic['coord']]
km.fit(enco)
reg.mapa_grupos(cov, km)

len(np.unique(dic.dic['coord']))
