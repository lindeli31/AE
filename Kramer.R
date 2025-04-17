library(tensorflow)
library(keras)
library(ggplot2)
library(dplyr)
set.seed(123)
theta = runif(100, 0, 2*pi)
y1 = 0.8*sin(theta)
y2 = 0.8*cos(theta)
dati = data.frame(y1 = y1, y2 = y2)
head(dati)
plot(dati$y1, dati$y2, asp = 1, pch = 20)

# pca ---------------------------------------------------------------------
#2 parametri liberi che sono i due loadings della prima componente principale

pca<- prcomp(dati)
#tenendo una sola componente principale
pc1<- pca$x[,1]
#andiamo a vedere l'errore di ricostruzione 
recpca <- pc1%*%t(pca$rotation[,1])
points(recpca, pch = 20, col = 'red')
#errore quadratico medio di ricostruzione 
sqrt(mean((as.matrix(dati)-recpca)^2))

# ANN no mapping layers ---------------------------------------------------
#7 parametri liberi di cui 2 bias input 1 bias bottleneck
#2 pesi da input a bottleneck 2 pesi d bottleneck a output






# NLPCA -------------------------------------------------------------------


model_nlpca<- keras_model_sequential() %>% 
  #strato input
  layer_dense(units = 4, input_shape = 2, activation="sigmoid") %>% 
  layer_dense(units = 1) %>% 
  layer_dense(units = 4, activation = "sigmoid")%>%
  layer_dense(units=2)
model_nlpca %>% compile(
  loss = "mse",
   optimizer = optimizer_adam(), 
) 
model_nlpca %>% fit(as.matrix(dati), as.matrix(dati), epochs = 100, batch_size = 10, validation_split = 0.2)
#val loss 0.1722,  loss 0.1573


# Predizioni (dati ricostruiti)
reconstructed_data_nlpca <- predict(model_nlpca, as.matrix(dati))
head(reconstructed_data_nlpca)


plot(dati$y1, dati$y2, asp = 1, pch = 16)
points(recpca, pch = 16, col = 'red')
# Visualizzazione dei dati originali e ricostruiti
points(reconstructed_data_nlpca, col = "steelblue", pch = 16)
legend("topright", legend = c("Dati Originali", "Dati Ricostruiti pca", "Dati ricostruiti AE"
                              ), col = c("black", "red", "steelblue"), pch = 16)



# coloro ------------------------------------------------------------------


