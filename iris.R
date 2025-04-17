library(keras)
library(dplyr)
library(tensorflow)
rm(list=ls())
# Carica e standardizza i dati
data(iris)
X <- iris %>% select(-Species) %>% scale()

# PCA
pca <- prcomp(X, rank = 2)
print(pca$rotation)
plot(pca$x[,1:2], col = iris$Species)
library(ggfortify)
biplot(pca)
autoplot(pca, data = iris, loadings = T, loadings.colour = "blue", 
         loadings.label = T, colour = "Species")

# Autoencoder
model <- keras_model_sequential() %>%
  layer_dense(units = 2, activation = 'linear', input_shape = 4) %>%
  layer_dense(units = 4, activation = 'linear')

model %>% compile(optimizer = optimizer_adam(0.01), loss = 'mse')
history <- model %>% fit(X, X, epochs = 1000, batch_size = 16, verbose = 0)
# Estrai pesi
W_encoder <- get_weights(model)[[1]]
print(W_encoder)


# Definisci un modello encoder per ottenere le proiezioni
encoder <- keras_model(inputs = model$input, outputs = get_layer(model, index = 1)$output)
ae_proj <- encoder %>% predict(X)

# Proiezione PCA
pca_proj <- pca$x

# Confronta le proiezioni (ad esempio, con una scatter plot)
plot(pca_proj[,1], pca_proj[,2], col = iris$Species, pch = 16, main = "Proiezione PCA")
plot(ae_proj[,1], ae_proj[,2], col = iris$Species, pch = 16, main = "Proiezione Autoencoder")

# Per un confronto quantitativo, ortonormalizza e confronta
Q_pca <- qr.Q(qr(pca$rotation))
Q_ae <- qr.Q(qr(W_encoder))
similarity <- t(Q_pca) %*% Q_ae
print(similarity)
# AE sparso ---------------------------------------------------------------


# Definizione del modello Sequential
model <- keras_model_sequential() %>%
  # Encoder con regolarizzazione L1 per sparsità
  layer_dense(units = 6, activation = "relu", input_shape = 4,
              activity_regularizer = regularizer_l1(0.01)) %>%  # Bottleneck overcomplete
  # Decoder
  layer_dense(units = 4, activation = "linear")  # Ricostruzione

# Compilazione
model %>% compile(
  optimizer = optimizer_adam(0.001),
  loss = "mse"
)

# Addestramento
history <- model %>% fit(
  X, X,
  epochs = 500,
  batch_size = 16,
  verbose = 0
)

# Estrazione dei codici latenti (bottleneck)
encoder <- keras_model(inputs = model$input, outputs = get_layer(model, index=1)$output)
encoded_data <- predict(encoder, X_scaled)
