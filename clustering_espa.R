setwd("D:/Codigos/COVID")

library(digest)
library(rgeoda)
library(sf)

fallec <- st_read('Geodabd/fallec.shp')
fallec_pca <- st_read('Geodabd/fallec_pca.shp')
fall_pos <- st_read('Geodabd/fall_pos.shp')
fall_pos_pca <- st_read('Geodabd/fall_pos_pca.shp')
positiv <- st_read('Geodabd/positiv.shp')
positiv_pca <- st_read('Geodabd/positiv_pca.shp')

w_knn6 <- knn_weights(fallec, 6)
bounds <- fallec['personas']
n_grupos <-  45

filas <- c('mes0','mes1','mes2','mes3','mes4','mes5','mes6','mes7','mes8','mes9','mes10','mes11','mes12','mes13','mes14','mes15','mes16')
filas2 <- c('mes0','mes1')#,'mes2','mes3')
filas3 <- c('mes0','mes1','mes2','mes3','mes4','mes5','mes6')
filas4 <- c('mes14')
guerr <- c('Crm_prs','Crm_prp','Litercy','Donatns','Infants','Suicids')
clusters <- function(df, nombre,filas){
datos <- df[filas]
skater <- skater(n_grupos, w_knn6, datos)
redcap <- redcap(n_grupos, w_knn6, datos, "fullorder-completelinkage")
schc <- schc(n_grupos,w_knn6,datos, 'complete')
#try(azpsa <- azp_sa(5,w_knn6, datos, cooling_rate = 0.85, random_seed = 456484))
#try(df['azpsa'] <- azpsa$Clusters)
#try(azptabu <- azp_tabu(5,w_knn6, datos,tabu_length = 10,conv_tabu = 10, random_seed = 456484))
#try(df['azptabu'] <- azptabu$Clusters)

try(maxpsa <- maxp_sa(w_knn6, datos,bound_variable = bounds,min_bound= 1000000, cooling_rate = 0.85,random_seed = 456484))
try(df['maxpsa'] <- maxpsa$Clusters)
#try(maxptabu <- maxp_tabu(w_knn6, datos,bound_variable = bounds,min_bound= 1000000 ,tabu_length = 10,conv_tabu = 10,random_seed = 456484))
#try(df['maxptabu'] <- maxptabu$Clusters)

df['sk'] <- skater$Clusters
df['redcap'] <- redcap$Clusters
df['schc'] <- schc$Clusters
direc <- paste('df_R/',nombre,'.shp',sep = '')
st_write(df,direc, delete_layer = T)}

t <- proc.time()
clusters(fallec, 'fallec',filas4)
clusters(fallec_pca, 'fallec_pca',filas2)
#clusters(positiv, 'positiv',filas)
#clusters(positiv_pca, 'positiv_pca',filas2)
#clusters(fall_pos, 'fall_pos',filas3)
#clusters(fall_pos_pca, 'fall_pos_pca',filas2)

dura <- proc.time()-t
dura
