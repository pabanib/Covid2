setwd("D:/Codigos/COVID")

library(digest)
library(rgeoda)
library(sf)


autoenco <- st_read('Geodabd/ae_mend.shp')
w_queen <- queen_weights(autoenco)

azptabu <- azp_tabu(31,w_queen, autoenco,tabu_length = 10,conv_tabu = 10, random_seed = 456484)
azpsa <- azp_sa(31,w_queen, autoenco, cooling_rate = 0.85, random_seed = 456484)

autoenco['ae_azp'] = azptabu$Clusters
autoenco['ae_azpsa'] = azpsa$Clusters

direc <- paste('df_R/','autoencod.shp',sep = '')
st_write(autoenco,direc, delete_layer = T)

