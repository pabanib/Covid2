setwd("D:/Archivos/Codigos/COVID")

library(digest)
library(rgeoda)
library(sf)

enco = st_read("Geodabd/sim4/encoders.shp")
prop = st_read("Geodabd/sim4/prop.shp")
prop3 = st_read("Geodabd/sim4/prop_3.shp")

datos <- enco[,c(names(enco)[1:6])]

w_knn6 <- knn_weights(enco, 6)
n_grupos = 6
skater <- skater(n_grupos, w_knn6, datos)
enco$sk <- skater$Clusters
plot(enco[c('sk','geometry')])

skater$`The ratio of between to total sum of squares`

redcap <- redcap(n_grupos, w_knn6, datos, "fullorder-completelinkage")     
enco$redcap <- redcap$Clusters
plot(enco[c('redcap','geometry')])

redcap$`The ratio of between to total sum of squares`

schc <- schc(n_grupos,w_knn6,datos, 'complete')
enco$schc <- schc$Clusters
plot(enco[c('schc','geometry')])

datos2 <- prop[,c(names(prop)[1:200])]

skater <- skater(n_grupos, w_knn6, datos2)
prop$sk <- skater$Clusters
plot(prop[c('sk','geometry')])

skater$`The ratio of between to total sum of squares`

redcap <- redcap(n_grupos, w_knn6, datos2)
prop$redcap <- redcap$Clusters
plot(prop[c('redcap','geometry')])

redcap$`The ratio of between to total sum of squares`


datos3 <- prop3[,c(names(prop3)[1:200])]

skater <- skater(n_grupos, w_knn6, datos3)
prop3$sk <- skater$Clusters
plot(prop3[c('sk','geometry')])

skater$`The ratio of between to total sum of squares`

redcap <- redcap(n_grupos, w_knn6, datos3)
prop3$redcap <- redcap$Clusters
plot(prop3[c('redcap','geometry')])

redcap$`The ratio of between to total sum of squares`
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


