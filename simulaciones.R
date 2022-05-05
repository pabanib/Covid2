#setwd("D:/Archivos/Codigos/COVID")
setwd("G:/My Drive/1. Tesis Pablo Quintana/covid/covid2")

library(digest)
library(rgeoda)
library(sf)

enco = st_read("Geodabd/sim4/encoders.shp")
prop = st_read("Geodabd/sim4/prop.shp")

rdos <- enco[,c(names(enco)[1:4])]

w_knn6 <- knn_weights(enco, 6)
n_grupos = 6
skater <- skater(n_grupos, w_knn6, enco)
rdos$ae_sk <- skater$Clusters
plot(rdos[c('ae_sk','geometry')])

skater$`The ratio of between to total sum of squares`

redcap <- redcap(n_grupos, w_knn6, enco, "fullorder-completelinkage")     
rdos$ae_redcap <- redcap$Clusters
plot(rdos[c('ae_redcap','geometry')])

redcap$`The ratio of between to total sum of squares`

schc <- schc(n_grupos,w_knn6,enco, 'complete')
rdos$ae_schc <- schc$Clusters
plot(rdos[c('ae_schc','geometry')])

# Todos los datos

skater <- skater(n_grupos, w_knn6, prop)
rdos$sk <- skater$Clusters
plot(rdos[c('sk','geometry')])

skater$`The ratio of between to total sum of squares`

redcap <- redcap(n_grupos, w_knn6, prop)
rdos$redcap <- redcap$Clusters
plot(rdos[c('redcap','geometry')])

redcap$`The ratio of between to total sum of squares`

st_write(rdos,"Geodabd/sim4/rdos_rgeod.shp", delete_layer = T)

######

enco = st_read("Geodabd/sim5/encoders.shp")
prop = st_read("Geodabd/sim5/prop.shp")

rdos <- enco[,c(names(enco)[1:4])]

w_knn6 <- knn_weights(enco, 6)
n_grupos = 6
skater <- skater(n_grupos, w_knn6, enco)
rdos$ae_sk <- skater$Clusters
plot(rdos[c('ae_sk','geometry')])

skater$`The ratio of between to total sum of squares`

redcap <- redcap(n_grupos, w_knn6, enco, "fullorder-completelinkage")     
rdos$ae_redcap <- redcap$Clusters
plot(rdos[c('ae_redcap','geometry')])

redcap$`The ratio of between to total sum of squares`

schc <- schc(n_grupos,w_knn6,enco, 'complete')
rdos$ae_schc <- schc$Clusters
plot(rdos[c('ae_schc','geometry')])

# Todos los datos

skater <- skater(n_grupos, w_knn6, prop)
rdos$sk <- skater$Clusters
plot(rdos[c('sk','geometry')])

skater$`The ratio of between to total sum of squares`

redcap <- redcap(n_grupos, w_knn6, prop)
rdos$redcap <- redcap$Clusters
plot(rdos[c('redcap','geometry')])

redcap$`The ratio of between to total sum of squares`

st_write(rdos,"Geodabd/sim5/rdos_rgeod.shp", delete_layer = T)

##### Tercera simulacion

enco = st_read("Geodabd/sim6/encoders.shp")
prop = st_read("Geodabd/sim6/prop.shp")

rdos <- enco[,c(names(enco)[1:4])]

w_knn6 <- knn_weights(enco, 6)
n_grupos = 6
skater <- skater(n_grupos, w_knn6, enco)
rdos$ae_sk <- skater$Clusters
plot(rdos[c('ae_sk','geometry')])

skater$`The ratio of between to total sum of squares`

redcap <- redcap(n_grupos, w_knn6, enco, "fullorder-completelinkage")     
rdos$ae_redcap <- redcap$Clusters
plot(rdos[c('ae_redcap','geometry')])

redcap$`The ratio of between to total sum of squares`

schc <- schc(n_grupos,w_knn6,enco, 'complete')
rdos$ae_schc <- schc$Clusters
plot(rdos[c('ae_schc','geometry')])

# Todos los datos

skater <- skater(n_grupos, w_knn6, prop)
rdos$sk <- skater$Clusters
plot(rdos[c('sk','geometry')])

skater$`The ratio of between to total sum of squares`

redcap <- redcap(n_grupos, w_knn6, prop)
rdos$redcap <- redcap$Clusters
plot(rdos[c('redcap','geometry')])

redcap$`The ratio of between to total sum of squares`

st_write(rdos,"Geodabd/sim6/rdos_rgeod.shp", delete_layer = T)


####

manifold = st_read("Geodabd/sim4/manifold.shp")
datos4 <- manifold[,c(names(manifold)[1:2])]

skater <- skater(n_grupos, w_knn6, manifold)
manifold$sk <- skater$Clusters
plot(manifold[c('sk','geometry')])

skater$`The ratio of between to total sum of squares`

redcap <- redcap(n_grupos, w_knn6, datos4, "fullorder-completelinkage")     
manifold$redcap <- redcap$Clusters
plot(manifold[c('redcap','geometry')])

redcap$`The ratio of between to total sum of squares`

### Caso real Covid19

enco = st_read("Geodabd/covid/encoders.shp")
prop = st_read("Geodabd/covid/prop.shp")

rdos <- enco[,c(names(enco)[1:4])]

w_knn6 <- knn_weights(enco, 6)
n_grupos = 6
skater <- skater(n_grupos, w_knn6, enco)
rdos$ae_sk <- skater$Clusters
plot(rdos[c('ae_sk','geometry')])

skater$`The ratio of between to total sum of squares`

redcap <- redcap(n_grupos, w_knn6, enco, "fullorder-completelinkage")     
rdos$ae_redcap <- redcap$Clusters
plot(rdos[c('ae_redcap','geometry')])

redcap$`The ratio of between to total sum of squares`

schc <- schc(n_grupos,w_knn6,enco, 'complete')
rdos$ae_schc <- schc$Clusters
plot(rdos[c('ae_schc','geometry')])

# Todos los datos

skater <- skater(n_grupos, w_knn6, prop)
rdos$sk <- skater$Clusters
plot(rdos[c('sk','geometry')])

skater$`The ratio of between to total sum of squares`

redcap <- redcap(n_grupos, w_knn6, prop)
rdos$redcap <- redcap$Clusters
plot(rdos[c('redcap','geometry')])

redcap$`The ratio of between to total sum of squares`

st_write(rdos,"Geodabd/covid/rdos_rgeod.shp", delete_layer = T)
