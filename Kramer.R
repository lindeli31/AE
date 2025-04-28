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
varianza_totale = sum(apply(dati, 2, var))
# pca ---------------------------------------------------------------------
#2 parametri liberi che sono i due loadings della prima componente principale

pca<- prcomp(dati, scale = F)
#tenendo una sola componente principale
#andiamo a vedere l'errore di ricostruzione 
pc1<- pca$x[,1]

recpca <- pc1%*%t(pca$rotation[,1])
#errore quadratico medio di ricostruzione 
E_pca <- mean((as.matrix(dati)-recpca)^2)/varianza_totale
E_pca
# ANN no mapping layers ---------------------------------------------------
#7 parametri liberi di cui 2 bias input 1 bias bottleneck
#2 pesi da input a bottleneck 2 pesi d bottleneck a output



# NLPCA ---------------------------------------------------------
#attivazioni tanh
# Normalizza in [-1, 1] (necessario per l'output lineare)
max_val <- max(abs(dati))
dati_scaled <- dati / max_val

model_nlpca <- keras_model_sequential() %>%
  # Mapping layer (primo hidden layer)
  layer_dense(units = 4, input_shape = 2, activation = "tanh") %>%
  # Bottleneck (secondo hidden layer)
  layer_dense(units = 1, activation = "linear") %>%
  # Demapping layer (terzo hidden layer)
  layer_dense(units = 4, activation = "tanh") %>%
  # Output layer (lineare per valori negativi)
  layer_dense(units = 2, activation = "linear")

model_nlpca %>% compile(
  loss = "mse",
  optimizer = optimizer_adam(learning_rate = 0.0001)
)

# Addestra il modello
history <- model_nlpca %>% fit(
  x = as.matrix(dati_scaled),
  y = as.matrix(dati_scaled),
  epochs = 5000,  # Più epoche per convergenza
  batch_size = 4, 
  verbose = 1
)

# Ricostruisco i dati e ri-scalo
recdata_nlpca <- predict(model_nlpca, as.matrix(dati_scaled)) * max_val
E_nlpca<- mean((as.matrix(dati)-recdata_nlpca)^2)
E_nlpca
E_pca
# Plot
plot(dati$y1, dati$y2, asp = 1, pch = 16, main = "NLPCA con 3 hidden layers")
points(recdata_nlpca, col = "red", pch = 16)
points(recpca, col = "purple", pch = 20)
legend("topright", legend = c("Originali", "Ricostruiti"), col = c("black", "red"), pch = 16)


