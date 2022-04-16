# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 19:18:35 2022

@author: Pablo
"""

from lectura_datos import *
import regionalizacion as reg
import matplotlib.pyplot as plt

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


cov =  covid.loc[pd.IndexSlice[:,'2021-01'],:].copy()
cov['reg'] = lista_regiones

var = cov.reg.map({1:0.0001,2:0.001,3:0.1, 4:0.3, 5:0.7})* cov.personas

regiones = cov.reg.copy()
regiones[cov.provincia == "Tucumán"] = 6
cov['reg'] = regiones
var = cov.reg.map({1:0.0001,2:0.001,3:0.1, 4:0.3, 5:0.7,6:0.7})* cov.personas
#transformo la variable en un GeoDataFrame para seguir trabajando y le agrego la variable personas que sería la variable poblacional
var = pd.concat([var,cov.personas], axis = 1)
var = var.rename(columns = {0:'var1'})
var = var.droplevel('mes')
var = gpd.GeoDataFrame(var, geometry = geo)
var.head()

periodos = 100
p = np.arange(0,periodos)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
pipe = Pipeline([('standard', StandardScaler())#,
                 #('normalizer', Normalizer('l1'))
                 ])


form = lambda x: ((x-50)/100)**2

func = {1 : lambda x : 0.14-((x-50)/100)**2,
     2 : lambda x : 0.25-((x-30)/100)**2,
     3 : lambda x : 0.20-((x-80)/100)**2,
     4 : lambda x : 0.10-((x-50)/100)**2,
     5 : lambda x : 0.3-((x-50)/100)**2,
     6 : lambda x : 0.3-((x-50)/100)**2
     }

for i in func.values():
    plt.plot(i(p))

func[1](p)

varianz = 0
x = cov.reg.map(func)
val = []
for i in x:
    y = i(p)+(np.random.rand()-0.5)*varianz
    val.append(y)
val = np.array(val)

var = val*cov[['personas']].values
var = pd.DataFrame(var, columns = pd.MultiIndex.from_tuples(tuple(zip(['var1']*periodos,p))), index = cov.index) 
var_ = var.stack()
var_['personas'] = np.array([cov.personas.values,]*periodos).T.reshape(525*periodos,1)
var = var_.droplevel('mes')
var = gpd.GeoDataFrame(var, geometry = np.array([cov.geometry.values,]*periodos).T.reshape(525*periodos,))
var.head()

sim3 = reg.entorno(var,['var1'],'personas', pipe)

from sklearn.cluster import KMeans, AgglomerativeClustering

sim3.procesar_datos()
param = {'n_clusters' : [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]} #se crean los distintos parametros para probar, en este caso solo de cantidad de grupos
metricas = {'hg': sim3.metric.Hg_relat, 'ind lq': sim3.metric.indice_lq} # se crean las metricas. En este caso las que provienen del LQ y es importante que salgan del entorno así buscan bien los datos
param_aglo = reg.copy(param)
param_aglo['connectivity'] = [sim3.W.sparse]

sim3.agregar_metodo('km', KMeans(), param, metricas)
sim3.agregar_metodo('aglo', AgglomerativeClustering(), param, metricas)
sim3.agregar_metodo('aglo_esp', AgglomerativeClustering(), param_aglo, metricas)

sim3.agregar_data('prop', val, ajustar = False)
#%%
def resultados(entorno):
    rdos = []
    for m in entorno.metodos_rdos.keys():
        r = []
        for n in entorno.metodos_rdos[m].keys():
            res = (entorno.metodos_rdos[m][n].best_metrics_).copy()
            res['Metodo'] = m
            res['Vers'] = n
            r.append(res[['Metodo','Vers','hg','ind lq']])
        rdos.extend(r)

    return pd.concat(rdos, axis = 0)



#%%
np.random.seed(3254)

lista = ['prop']
sim3.calcular_metodo('km', lista)
sim3.calcular_metodo('aglo', lista)
sim3.calcular_metodo('aglo_esp', lista)
sim3.mapa('km'),sim3.mapa('aglo'),sim3.mapa('aglo_esp')

#%%
#sin centroides

np.random.seed(32)
sim3.calcular_metodo('km', lista, centroides = False)
sim3.calcular_metodo('aglo', lista, centroides = False)
sim3.calcular_metodo('aglo_esp', lista, centroides = False)
sim3.mapa('km'),sim3.mapa('aglo'),sim3.mapa('aglo_esp')

#%%
# autoencoders
np.random.seed(32)
lista = ['prop']
n_enc = 6
sim3.calcular_metodo('km', lista, ae = True, n_encoders =6, centroides = False)
sim3.calcular_metodo('aglo', lista,ae = True, n_encoders =6, centroides = False)
sim3.calcular_metodo('aglo_esp', lista,ae = True, n_encoders =6, centroides = False)
sim3.mapa('km'),sim3.mapa('aglo'),sim3.mapa('aglo_esp')


geoenco = reg.gpd.GeoDataFrame(sim3.enco.predict(sim3.retornar_dfs(separado = lista)), columns = ['enco'+str(i) for i in range(n_enc)], geometry = sim3.geo.values)
geoprop = reg.gpd.GeoDataFrame(sim3.retornar_dfs(separado = lista)[0],columns = ['peri' + str(i) for i in p] ,geometry = sim3.geo.values)

geoenco.to_file("Geodabd/sim3/encoders.shp")
geoprop.to_file("Geodabd/sim3/prop.shp")

#%%

sim3.agregar_data('minmax',np.c_[sim3.dic['prop'].max(axis = 1),sim3.dic['prop'].min(axis = 1)],ajustar = False)

lista = [['prop'],'minmax']
n_enc = 6
sim3.calcular_metodo('km', lista, ae = True, n_encoders =6, centroides = False)
sim3.calcular_metodo('aglo', lista,ae = True, n_encoders =6, centroides = False)
sim3.calcular_metodo('aglo_esp', lista,ae = True, n_encoders =6, centroides = False)
sim3.mapa('km'),sim3.mapa('aglo'),sim3.mapa('aglo_esp')

geoenco = reg.gpd.GeoDataFrame(sim3.enco.predict(sim3.retornar_dfs(separado = lista)), columns = ['enco'+str(i) for i in range(n_enc)], geometry = sim3.geo.values)
geoenco.to_file("Geodabd/sim3/encoders.shp")

#%%


func2 = {1 : lambda x : 0.14-((2*x-90)/160)**2,
     2 : lambda x : 0.25-((2*x-100)/160)**2,
     3 : lambda x : 0.20-((2*x-150)/160)**2,
     4 : lambda x : 0.20-((2*x-100)/160)**2,
     5 : lambda x : 0.15-((2*x-120)/160)**2,
     6 : lambda x : 0.3-((2*x-120)/160)**2
     }

for i in func2.values():
    plt.plot(i(p)  +np.random.randn(len(p))*0.05)*varianz

x2 = cov.reg.map(func2)
val2 = []
for i in x2:
    y = i(p)+(np.random.rand()-0.5)*varianz
    val2.append(y)
val2 = np.array(val2)

val
val2

v = np.c_[(val * (val > 0)),(val2 * (val2 > 0))] 

var = v*cov[['personas']].values
l = list(zip(['var1']*periodos,p))
l.extend(list(zip(['var2']*periodos,p)))
l = tuple(l)
var = pd.DataFrame(var, columns = pd.MultiIndex.from_tuples(l), index = cov.index) 
var_ = var.stack()
var_['personas'] = np.array([cov.personas.values,]*periodos).T.reshape(525*periodos,1)
var = var_.droplevel('mes')
var = gpd.GeoDataFrame(var, geometry = np.array([cov.geometry.values,]*periodos).T.reshape(525*periodos,))
var.head()

sim4 = reg.entorno(var,['var1','var2'],'personas', pipe)
sim4.procesar_datos()
param = {'n_clusters' : [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]} #se crean los distintos parametros para probar, en este caso solo de cantidad de grupos
metricas = {'hg': sim3.metric.Hg_relat, 'ind lq': sim3.metric.indice_lq} # se crean las metricas. En este caso las que provienen del LQ y es importante que salgan del entorno así buscan bien los datos
param_aglo = reg.copy(param)
param_aglo['connectivity'] = [sim3.W.sparse]

sim4.agregar_metodo('km', KMeans(), param, metricas)
sim4.agregar_metodo('aglo', AgglomerativeClustering(), param, metricas)
sim4.agregar_metodo('aglo_esp', AgglomerativeClustering(), param_aglo, metricas)

sim4.agregar_data('prop', v, ajustar = False)

#%%
np.random.seed(3254)

lista = ['prop']
sim4.calcular_metodo('km', lista)
sim4.calcular_metodo('aglo', lista)
sim4.calcular_metodo('aglo_esp', lista)
sim4.mapa('km'),sim4.mapa('aglo'),sim4.mapa('aglo_esp')

#%%
np.random.seed(3254)

lista = ['prop']
sim4.calcular_metodo('km', lista, centoides = False)
sim4.calcular_metodo('aglo', lista, centroides = False)
sim4.calcular_metodo('aglo_esp', lista, centroides = False)
sim4.mapa('km'),sim4.mapa('aglo'),sim4.mapa('aglo_esp')

#%%
np.random.seed(3254)

lista = ['prop']
n_enc = 6
sim4.calcular_metodo('km', lista, centroides = False,ae = True, n_encoders = n_enc)
sim4.calcular_metodo('aglo', lista, centroides = False,ae = True, n_encoders = n_enc)
sim4.calcular_metodo('aglo_esp', lista, centroides = False,ae = True, n_encoders = n_enc)
sim4.mapa('km'),sim4.mapa('aglo'),sim4.mapa('aglo_esp')


geoenco = reg.gpd.GeoDataFrame(sim4.enco.predict(sim4.retornar_dfs(separado = lista)), columns = ['enco'+str(i) for i in range(n_enc)], geometry = sim4.geo.values)
geoprop = reg.gpd.GeoDataFrame(sim4.retornar_dfs(separado = lista)[0],columns = [l[i][0]+str(l[i][1]) for i in range(len(l))] ,geometry = sim4.geo.values)

geoenco.to_file("Geodabd/sim4/encoders.shp")
geoprop.to_file("Geodabd/sim4/prop.shp")
#%%

sim4.agregar_data('W', sim4.W_queen.sparse.todense(), ajustar = False)
n_enc = 6
lista = [['prop'],'W']
sim4.calcular_metodo('km', lista, centroides = False,ae = True, n_encoders = n_enc, optimizer = 'adam', loss = 'cosine_similarity')
sim4.calcular_metodo('aglo', lista, centroides = False,ae = True, n_encoders = n_enc, optimizer = 'adam', loss = 'cosine_similarity')
sim4.calcular_metodo('aglo_esp', lista, centroides = False,ae = True, n_encoders = n_enc, optimizer = 'adam', loss = 'cosine_similarity')
sim4.mapa('km'),sim4.mapa('aglo'),sim4.mapa('aglo_esp')


geoenco = reg.gpd.GeoDataFrame(sim4.enco.predict(sim4.retornar_dfs(separado = lista)), columns = ['enco'+str(i) for i in range(n_enc)], geometry = sim4.geo.values)
geoprop = reg.gpd.GeoDataFrame(sim4.retornar_dfs(separado = lista)[0],columns = [l[i][0]+str(l[i][1]) for i in range(len(l))] ,geometry = sim4.geo.values)

geoenco.to_file("Geodabd/sim4/encoders.shp")
geoprop.to_file("Geodabd/sim4/prop.shp")

#%%
sim4.agregar_data('prop1', sim4.dic['prop'][:,:periodos], ajustar = False)
sim4.agregar_data('prop2', sim4.dic['prop'][:,periodos:], ajustar = False)

lista = ['prop1','prop2']
n_enc = 4

sim4.calcular_metodo('km', lista, centroides = False,ae = True, n_encoders = n_enc, optimizer = 'adam', loss = 'cosine_similarity')
sim4.calcular_metodo('aglo', lista, centroides = False,ae = True, n_encoders = n_enc, optimizer = 'adam', loss = 'cosine_similarity')
sim4.calcular_metodo('aglo_esp', lista, centroides = False,ae = True, n_encoders = n_enc, optimizer = 'adam', loss = 'cosine_similarity')
sim4.mapa('km'),sim4.mapa('aglo'),sim4.mapa('aglo_esp')

geoenco = reg.gpd.GeoDataFrame(sim4.enco.predict(sim4.retornar_dfs(separado = lista)), columns = ['enco'+str(i) for i in range(n_enc)], geometry = sim4.geo.values)
geoenco.to_file("Geodabd/sim4/encoders.shp")

#%%
from sklearn import manifold
n_enc = 4
sim4.agregar_data('manifold'
                  ,manifold.SpectralEmbedding(n_components=n_enc).fit_transform(sim4.retornar_dfs(separado = [lista])[0])
                  ,ajustar = False)


sim4.calcular_metodo('km', ['manifold'], centoides = False)
sim4.calcular_metodo('aglo', ['manifold'], centroides = False)
sim4.calcular_metodo('aglo_esp', ['manifold'], centroides = False)
sim4.mapa('km'),sim4.mapa('aglo'),sim4.mapa('aglo_esp')


geomani = reg.gpd.GeoDataFrame(sim4.retornar_dfs(separado = ['manifold'])[0], columns = ['enco'+str(i) for i in range(n_enc)], geometry = sim4.geo.values)
geomani.to_file("Geodabd/sim4/manifold.shp")

#%%

sim4.interacciones()


#%%
n_enc = 6

inp1 = reg.layers.Input(shape = [100,])
inp2 = reg.layers.Input(shape = [100,])
dense1 = reg.layers.Dense(80, activation = "relu")(inp1)
dense1 = reg.layers.Dense(40, activation = "relu")(dense1)
dense1 = reg.layers.Dense(3, activation = "relu")(dense1)
dense2 = reg.layers.Dense(80, activation = "relu")(inp2)
dense2 = reg.layers.Dense(40, activation = "relu")(dense2)
dense2 = reg.layers.Dense(3, activation = "relu")(dense2)
concat = reg.layers.concatenate([dense1,dense2])
concat = reg.layers.Dense(50, activation = "relu")(concat)
enco = reg.layers.Dense(n_enc, activation = "relu")(concat)
dense = reg.layers.Dense(40, activation = "relu",  kernel_regularizer = reg.regularizers.l2(0.01))(enco)
dense = reg.layers.Dense(80, activation = "relu",  kernel_regularizer = reg.regularizers.l2(0.01))(dense)
decoder = reg.layers.Dense(200, activation = "relu",  kernel_regularizer = reg.regularizers.l2(0.01))(dense)

autoencoder = reg.Model(inputs = [inp1,inp2], outputs = decoder)
encoder = reg.Model(inputs = [inp1,inp2], outputs = enco)
#%%
autoencoder.compile(optimizer = 'adam', loss = 'cosine_similarity')

history = autoencoder.fit(sim4.retornar_dfs(separado = lista),sim4.retornar_dfs(separado = [lista]), epochs = 100, validation_split = 0.2 , verbose = False) 
history = pd.DataFrame(history.history)
history.loc[:,["loss","val_loss"]].plot()
print("Minimum validation loss: {}".format(history['val_loss'].min()))
#%%
sim4.agregar_data('enco', encoder.predict(sim4.retornar_dfs(separado = lista)), ajustar = False)

sim4.calcular_metodo('km', ['enco'], centoides = False)
sim4.calcular_metodo('aglo', ['enco'], centroides = False)
sim4.calcular_metodo('aglo_esp', ['enco'], centroides = False)
sim4.mapa('km'),sim4.mapa('aglo'),sim4.mapa('aglo_esp')



geoenco = reg.gpd.GeoDataFrame(encoder.predict(sim4.retornar_dfs(separado = lista)), columns = ['enco'+str(i) for i in range(n_enc)], geometry = sim4.geo.values)
geoenco.to_file("Geodabd/sim4/encoders.shp")


#%%
from deep_cluster import ClusteringLayer
#%%
n_clusters = 6
clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
model = reg.Model(inputs=encoder.input, outputs=clustering_layer)
model.compile(optimizer='adam', loss='kld')
model.Trainable = True
#model.best_model_['modelo'].iloc[0].labels_
#sim4.calcular_metodo('km', lista, centroides = True,ae = True, n_encoders = 6)
#sim4.mapa('km')
#km = sim4.Metodos['km'].modelos.iloc[4]['modelo']
from sklearn.cluster import KMeans, AgglomerativeClustering

km = KMeans(6)
km.fit(sim4.dic['enco'])
cc = km.cluster_centers_
y_pred = km.labels_
y_pred_last = np.copy(y_pred)
model.get_layer(name="clustering").set_weights([cc])

def target_distribution(q):
    weight = (q ** 2 / q.sum(0))
    return (weight.T / weight.sum(1).reshape(525,)).T

loss = 0
index = 0
maxiter = 1000
update_interval = 20
index_array = np.arange(x.shape[0])
tol = 0.0001 # tolerance threshold to stop training
#%%
X = sim4.retornar_dfs(separado = lista)
aglo = AgglomerativeClustering(n_clusters, connectivity=sim4.W_queen.sparse )

batch_size = 70
for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q = model.predict(X, verbose=0)
        p = target_distribution(q)  # update the auxiliary target distribution p

        # evaluate the clustering performance
        y_pred = aglo.fit_predict(q)
        #y_pred = q.argmax(1)
        
        """
        if y is not None:
            acc = 0 #np.round(metrics.acc(y, y_pred), 5)
            nmi = np.round(metrics.nmi(y, y_pred), 5)
            ari = np.round(metrics.ari(y, y_pred), 5)
            loss = np.round(loss, 5)
            print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)     """
        # check stop criterion - model convergence
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    idx = index_array[index * batch_size: min((index+1) * batch_size, X[0].shape[0])]
    #loss = model.train_on_batch(x=[X[0][idx],X[1][idx]], y=p[idx])
    loss = model.fit(x=[X[0],X[1]], y=p)
    index = index + 1 if (index + 1) * batch_size <= X[0].shape[0] else 0

q = model.predict(X)
p = target_distribution(q)
#y_pred = q.argmax(1)
y_pred = aglo.fit_predict(q)

#%%

sim4.df.plot(y_pred, categorical = True)

#%%
np.random.seed(456)
sim4.agregar_data('prop3',(sim4.dic['prop1']+sim4.dic['prop2']+np.random.rand(525,100))/3,ajustar = False)

lista = ['prop1','prop2','prop3']
n_enc = 2

sim4.calcular_metodo('km', lista, centroides = False,ae = True, n_encoders = n_enc, optimizer = 'adam', loss = 'cosine_similarity')
sim4.calcular_metodo('aglo', lista, centroides = False,ae = True, n_encoders = n_enc, optimizer = 'adam', loss = 'cosine_similarity')
sim4.calcular_metodo('aglo_esp', lista, centroides = False,ae = True, n_encoders = n_enc, optimizer = 'adam', loss = 'cosine_similarity')
sim4.mapa('km'),sim4.mapa('aglo'),sim4.mapa('aglo_esp')

geoenco = reg.gpd.GeoDataFrame(sim4.enco.predict(sim4.retornar_dfs(separado = lista)), columns = ['enco'+str(i) for i in range(n_enc)], geometry = sim4.geo.values)
geoenco.to_file("Geodabd/sim4/encoders.shp")

#%%

l = list(zip(['var1']*periodos,p))
l.extend(list(zip(['var2']*periodos,p)))
l.extend(list(zip(['var3']*periodos,p)))
l = tuple(l)

geoprop = reg.gpd.GeoDataFrame(sim4.retornar_dfs(separado = [lista])[0],columns = [l[i][0]+str(l[i][1]) for i in range(len(l))] ,geometry = sim4.geo.values)
geoprop.to_file("Geodabd/sim4/prop_3.shp")

#%%

from sklearn import manifold

sim4.agregar_data('manifold'
                  ,manifold.SpectralEmbedding(n_components=2).fit_transform(sim4.retornar_dfs(separado = [lista])[0])
                  ,ajustar = False)


sim4.calcular_metodo('km', ['manifold'], centoides = False)
sim4.calcular_metodo('aglo', ['manifold'], centroides = False)
sim4.calcular_metodo('aglo_esp', ['manifold'], centroides = False)
sim4.mapa('km'),sim4.mapa('aglo'),sim4.mapa('aglo_esp')


geomani = reg.gpd.GeoDataFrame(sim4.retornar_dfs(separado = ['manifold'])[0], columns = ['enco'+str(i) for i in range(n_enc)], geometry = sim4.geo.values)
geomani.to_file("Geodabd/sim4/manifold.shp")

