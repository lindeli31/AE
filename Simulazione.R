#################################################################
# SCRIPT PER STUDIO DI SIMULAZIONE: PCA vs FA vs AE
#################################################################

# --- 1. LIBRERIE E SETUP ---------------------------------------
# Assicurati di aver installato tutti questi pacchetti
# install.packages(c("MASS", "psych", "keras", "dplyr", "purrr", "pracma", "clue"))
library(MASS)        # Per mvrnorm
library(psych)       # Per la Factor Analysis (fa)
library(keras)
library(tensorflow)
library(dplyr)       # Per la manipolazione dei dati (tibble)
library(purrr)       # Utile per la programmazione funzionale
library(pracma)      # Per il problema di Procrustes (opzionale, ma utile per capire)
library(clue)        # Per solve_LSAP in fact_cor
library(ggplot2)
library(tidyr)
library(knitr)
library(gtools)
library(ks)   

callback_list <- list(
  callback_early_stopping(
    monitor = "val_loss",
    patience = 40,
    min_delta = 0.0001,      # Miglioramento minimo considerato significativo
    mode = "min",             # Cerca la diminuzione della loss
    restore_best_weights = TRUE
  ),
  callback_reduce_lr_on_plateau(
    monitor = "val_loss",
    factor = 0.5,
    patience = 20,
    min_delta = 0.001,       # Soglia per considerare un plateau
    min_lr = 1e-7,           # Learning rate minimo
    verbose = 10
  )
)


# --- 2. FUNZIONI DI SUPPORTO E METRICHE (Riscritte) --------------

# una funzione unica che calcola tutte le metriche per
# coerenza e efficienza
calculate_all_metrics <- function(Z_true, Z_estimated) {
  if (any(is.na(Z_estimated))) {
    return(list(factor_cor = NA, recon_err_permuted = NA, recon_err_rotated = NA))
  }
  
  # --- 1. GESTIONE PERMUTAZIONE E SEGNO ---
  C <- abs(cor(Z_true, Z_estimated))
  perm <- clue::solve_LSAP(C, maximum = TRUE)
  Z_permuted <- Z_estimated[, perm]
  signs <- sign(diag(cor(Z_true, Z_permuted)))
  Z_aligned <- sweep(Z_permuted, 2, signs, FUN = "*")
  
  # --- 2. CALCOLO METRICHE DI PERFORMANCE ---
  # Metrica 1: Correlazione Fattoriale
  factor_cor_val <- mean(diag(cor(Z_true, Z_aligned))^2)
  
  # Metrica 2: Errore Post-Permutazione (sostituisce l'errore grezzo)
  recon_err_permuted_val <- mean((Z_true - Z_aligned)^2)
  
  # --- 3. GESTIONE ROTAZIONE (PROCRUSTES) ---
  svd_results <- svd(t(Z_estimated) %*% Z_true)
  R <- svd_results$v %*% t(svd_results$u)
  Z_rotated <- Z_estimated %*% R
  recon_err_rotated_val <- mean((Z_true - Z_rotated)^2)
  
  return(list(
    factor_cor = factor_cor_val,
    recon_err_permuted = recon_err_permuted_val,
    recon_err_rotated = recon_err_rotated_val
  ))
}

#funzione per generare la matrice dei pesi sparsa
#usando la distribuzione di dirichlet 
generate_sparse_weights_controlled <- function(d, p,
                                               alpha_val = 0.1,
                                               strong_indicator_proportion = 0.75,
                                               strong_magnitude_range = c(3, 7),
                                               weak_magnitude_range = c(0.5, 1.5)) {
  
  # 1. Calcola il numero di indicatori forti e deboli
  num_strong <- floor(p * strong_indicator_proportion)
  num_weak <- p - num_strong
  
  # 2. Assegna casualmente i ruoli (forte/debole) alle 'p' variabili
  #    Creo un vettore di indici da 1 a p e lo mescolo.
  shuffled_indices <- sample(1:p)
  strong_indices <- shuffled_indices[1:num_strong]
  weak_indices <- if (num_weak > 0) shuffled_indices[(num_strong + 1):p] else integer(0)
  
  # Crea la matrice vuota per i pesi
  W <- matrix(0, nrow = d, ncol = p)
  
  # Itera su ogni variabile osservata (colonna di W)
  for (j in 1:p) {
    # 3. Genera la struttura sparsa interna (invariata)
    proportions <- t(rdirichlet(1, rep(alpha_val, d)))
    signs <- sample(c(-1, 1), size = d, replace = TRUE)
    
    # 4. Determina la magnitudo totale in base al ruolo della variabile
    if (j %in% strong_indices) {
      # Questa e' una variabile "forte", campiono dal range forte
      total_magnitude <- runif(1, min = strong_magnitude_range[1], max = strong_magnitude_range[2])
    } else {
      # Questa e' una variabile "debole", campiono dal range debole
      total_magnitude <- runif(1, min = weak_magnitude_range[1], max = weak_magnitude_range[2])
    }
    
    # 5. Combina tutto per creare la colonna j-esima dei pesi
    W[, j] <- signs * proportions * total_magnitude
  }
  
  return(W)
}

# --- 3. FUNZIONE DI SIMULAZIONE PRINCIPALE  ----------

run_single_simulation <- function(n, d, p, noise_level, transformation, replication_id, W) {
  
  # ... (La parte A e B di generazione dati e training dei modelli rimane identica) ...
  set.seed(n + d*10 + p*100 + replication_id*1000 + ifelse(noise_level=="low", 0, 5000) + ifelse(transformation=="linear", 0, 10000))
  tf$random$set_seed(n + d*10 + p*100 + replication_id*1000)
  Z <- matrix(rnorm(n * d), nrow = n, ncol = d); X <- matrix(0, nrow = n, ncol = p)
  if (transformation == "linear") { X <- Z %*% W } 
  else {
    # --- NUOVE TRASFORMAZIONI: Z^3 e Exp(Z) con Standardizzazione ---
    p_part1 <- p / 2
    
    # 1. Applica le trasformazioni non-lineari base
    Z_part1_transformed <- Z^3
    Z_part2_transformed <- exp(0.5*Z)
    
    # 2. Standardizza l'output per controllare la varianza
    Z_part1_std <- scale(Z_part1_transformed)
    Z_part2_std <- scale(Z_part2_transformed)
    
    # 3. Genera il segnale non-lineare
    X[, 1:p_part1]     <- Z_part1_std %*% W$part1
    X[, (p_part1 + 1):p] <- Z_part2_std %*% W$part2
  }
  noise_sd_val <- ifelse(noise_level == "low", 0.1, 0.5)
  noise <- matrix(rnorm(n * p, mean = 0, sd = noise_sd_val * sd(as.vector(X))), nrow = n, ncol = p)
  X_final <- X + noise
  X_scaled <- scale(X_final)
  pca_model <- prcomp(X_scaled, center = FALSE, scale. = FALSE)
  message("Completato PCA\n")
  Z_pca <- pca_model$x[, 1:d]
  fa_model <- tryCatch({ psych::fa(X_scaled, nfactors = d, rotate = "varimax", fm = "ml", scores = "regression") }, error = function(e) { NULL })
  message("Completato FA\n")
  Z_fa <- if (!is.null(fa_model)) fa_model$scores else matrix(NA, nrow = n, ncol = d)
  k_clear_session()
  units_l1 <- min(1024, round(p * 1.2)) 
  compression_ratio <- (d / units_l1)^(1/3)
  units_l2_ideal <- round(units_l1 * compression_ratio)
  units_l3_ideal <- round(units_l2_ideal * compression_ratio)
  units_l3 <- max(units_l3_ideal, d + 1)
  units_l2 <- max(units_l2_ideal, units_l3 + 1)
  if (units_l1 <= units_l2) { units_l1 <- units_l2 + 1 }
  
  
  ae_model <- keras_model_sequential(name = "Autoencoder") %>%
    layer_dense(units=units_l1, activation="tanh", input_shape=ncol(X_scaled), kernel_initializer=initializer_glorot_normal()) %>%
    layer_dense(units=units_l2, activation="tanh", kernel_initializer=initializer_glorot_normal()) %>%
    layer_dense(units=units_l3, activation="tanh", kernel_initializer=initializer_glorot_normal()) %>%
    layer_dense(units=d, name="code", activation="linear") %>%
    layer_dense(units=units_l3, activation="tanh", kernel_initializer=initializer_glorot_normal()) %>%
    layer_dense(units=units_l2, activation="tanh", kernel_initializer=initializer_glorot_normal()) %>%
    layer_dense(units=units_l1, activation="tanh", kernel_initializer=initializer_glorot_normal()) %>%
    layer_dense(units=ncol(X_scaled), activation="linear")

  history<- ae_model %>% compile(optimizer="adam", loss="mse"); ae_model %>% fit(X_scaled, X_scaled, epochs=1000, batch_size=64, callbacks=callback_list, validation_split=0.2, verbose=0)
  encoder_model <- keras_model(inputs=ae_model$input, outputs=get_layer(ae_model, "code")$output); Z_ae <- predict(encoder_model, X_scaled)
  message("Completato AE\n")
  # C. CALCOLO METRICHE (Riscritto)
  metrics_pca <- calculate_all_metrics(Z, Z_pca)
  metrics_fa  <- calculate_all_metrics(Z, Z_fa)
  metrics_ae  <- calculate_all_metrics(Z, Z_ae)
  message("Completato metriche\n")
  
  
  # MODIFICA: Metriche sullo spazio dati X
  # Per AE: prendo la miglior val_loss, che e' l'MSE sul validation set.
  data_recon_mse_ae <- min(history$metrics$val_loss, na.rm = TRUE)
  
  # Per PCA: ricostruisco X_scaled e calcolo l'MSE
  X_recon_pca <- Z_pca %*% t(pca_model$rotation[, 1:d])
  data_recon_mse_pca <- mean((X_scaled - X_recon_pca)^2)
  
  # Per FA: ricostruisco X_scaled e calcolo l'MSE
  data_recon_mse_fa <- if (!is.null(fa_model)) {
    X_recon_fa <- Z_fa %*% t(fa_model$loadings)
    mean((X_scaled - X_recon_fa)^2)
  } else {
    NA
  }
  
  # Consolidamento di tutti i risultati
  results_tibble <- tibble(
    n = n, d = d, p = p, noise_level = noise_level, transformation = transformation, rep_id = replication_id,
    method = c("PCA", "FA", "AE"),
    factor_cor = c(metrics_pca$factor_cor, metrics_fa$factor_cor, metrics_ae$factor_cor),
    recon_err_permuted = c(metrics_pca$recon_err_permuted, metrics_fa$recon_err_permuted, metrics_ae$recon_err_permuted),
    recon_err_rotated = c(metrics_pca$recon_err_rotated, metrics_fa$recon_err_rotated, metrics_ae$recon_err_rotated),
    data_recon_mse = c(data_recon_mse_pca, data_recon_mse_fa, data_recon_mse_ae)
  )
  
  # Restituisco una lista con risultati e scores grezzi
  output_data <- list(
    results = results_tibble,
    scores = list(pca = Z_pca, fa = Z_fa, ae = Z_ae)
  )
  return(output_data)
}



# --- 4. SCRIPT DI ORCHESTRAZIONE  -------------------

n_replications <-50
output_dir <- "risultati_simulazione3" 

params_unique <- expand.grid(n=c(1000, 10000),d=c(2, 5),p=c(10, 100, 1000),
                             noise_level=c("low", "high"),
                             transformation=c( "linear", "nonlinear"),stringsAsFactors=FALSE)

for (i in 1:nrow(params_unique)) {
  current_params <- params_unique[i, ]
  
  # ... (log di avanzamento e generazione di W ) ...
  message(sprintf("--- INIZIO SET DI PARAMETRI %d/%d: n=%d, d=%d, ...", 
                  i, nrow(params_unique), current_params$n, current_params$d))
  set.seed(current_params$n+current_params$d*10+current_params$p*100+
             ifelse(current_params$transformation=="linear",0,10000))
  # Chiamo la nuova funzione controllata. I parametri come la proporzione 75/25
  # sono gia' impostati di default nella funzione.
  # 1. Genero la matrice di pesi sparsa e controllata.
  W_sparse <- generate_sparse_weights_controlled(
    d = current_params$d, 
    p = current_params$p
  )
  
  # 2. Assegno questa matrice (o le sue parti) a W, a seconda dello scenario.
  # prima del ciclo for per le replicazioni fisso la matrice dei pesi 
  if (current_params$transformation == "linear") {
    # Nello scenario lineare, uso direttamente la matrice sparsa.
    W <- W_sparse
  } else { # Scenario non-lineare
    p_part1 <- current_params$p / 2
    
    W <- list(
      part1 = W_sparse[, 1:p_part1],
      part2 = W_sparse[, (p_part1 + 1):current_params$p]
    )
  }
 
  
  for (rep_id in 1:n_replications) {
    message(sprintf("  -> Avvio replicazione %d/%d...", rep_id, n_replications))
    
    # Eseguo la simulazione
    run_output <- run_single_simulation(
      n = current_params$n, d = current_params$d, p = current_params$p,
      noise_level = current_params$noise_level, transformation = current_params$transformation,
      replication_id = rep_id, W = W
    )
    
    # Salvo il risultato di questa specifica run
    file_name <- sprintf("%s/res_n%d_d%d_p%d_%s_%s_rep%d.RData",
                         output_dir, current_params$n, current_params$d, current_params$p,
                         current_params$noise_level, current_params$transformation, rep_id)
    
    ### MODIFICA 3: Salvo l'intero oggetto 'run_output'
    save(run_output, file = file_name)
    
  }
  
  message(sprintf("--- FINE SET DI PARAMETRI %d/%d. ---", i, nrow(params_unique)))
}





# analisi -----------------------------------------------------------------



# --- 1. CARICAMENTO DATI ---

output_dir <- "risultati_simulazione3" 

file_list <- list.files(path = output_dir, pattern = "\\.RData$", full.names = TRUE)

all_results <- map_dfr(file_list, function(file_path) {
  env <- new.env()
  load(file_path, envir = env)
  return(env$run_output$results)
}, .progress = TRUE)


# ---. VERIFICA E PRIMA ISPEZIONE ---

total_combinations <- 2 * 2 * 3 * 2 * 2 * 50
expected_rows <- total_combinations * 3
cat(paste("Numero totale di righe nel dataframe 'all_results':", nrow(all_results), "\n"))
cat(paste("Numero di righe attese:", expected_rows, "\n"))
if (nrow(all_results) != expected_rows) {
  warning("ATTENZIONE: Il numero di righe non corrisponde a quello atteso.")
}
glimpse(all_results)



#  ---VISUALIZZAZIONI ---

# --- SETUP GRAFICI ---
theme_set(
  theme_bw() +
    theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 11, hjust = 0.5),
      legend.position = "top",
      strip.text = element_text(face = "bold", size = 9),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
)
method_colors <- c("AE" = "#1f77b4", "FA" = "#ff7f0e", "PCA" = "#2ca02c")

all_results<- all_results %>%
  mutate(
    # Usa case_when() per tradurre i valori in una nuova colonna
    livello_rumore = case_when(
      noise_level == "low"  ~ "Basso",
      noise_level == "high" ~ "Alto",
      TRUE                  ~ noise_level # Lascia invariati altri valori se ce ne fossero
    ),
    # Converte la nuova colonna in un fattore per garantire l'ordine corretto ("Basso" prima di "Alto")
    livello_rumore = factor(livello_rumore, levels = c("Basso", "Alto"))
  )
# --- GRAFICO 1: Correlazione Fattoriale Media (factor_cor) ---

# MIO COMMENTO: Questo grafico è corretto e rimane invariato.
plot_factor_cor_linear <- all_results %>%
  filter(transformation == "linear") %>% 
  ggplot(aes(x = as.factor(n), y = factor_cor, fill = method)) +
  geom_boxplot(na.rm = TRUE, alpha = 0.8, outlier.shape = NA) +
  facet_grid(rows = vars(livello_rumore), cols = vars(p, d), labeller = label_both) +
  scale_fill_manual(values = method_colors) + ylim(0, 1.05) +
  labs( y = "Correlazione media", x = "Numero di Osservazioni (n)")+
  theme(legend.position = "none")

plot_factor_cor_nonlinear <- all_results %>%
  filter(transformation == "nonlinear") %>% 
  ggplot(aes(x = as.factor(n), y = factor_cor, fill = method)) +
  geom_boxplot(na.rm = TRUE, alpha = 0.8, outlier.shape = NA) +
  facet_grid(rows = vars(livello_rumore), cols = vars(p, d), labeller = label_both) +
  scale_fill_manual(values = method_colors) + ylim(0, 1.05) +
  labs( y = "Correlazione media", x = "Numero di Osservazioni (n)")+
  theme(legend.position = "none")

print(plot_factor_cor_linear)
print(plot_factor_cor_nonlinear)


# --- GRAFICO 2: Errore Geometrico Post-Rotazione (Distanza di Procruste) ---


cat("\n--- Creazione Grafico Errore Geometrico Post-Rotazione (Procruste) ---\n")

# --- GRAFICO LINEARE ---
plot_procrustes_error_linear <- all_results %>%
  filter(transformation == "linear") %>%
  # MODIFICA: Ora usiamo 'recon_err_rotated' per tutti i metodi sull'asse y
  ggplot(aes(x = as.factor(n), y = recon_err_rotated, fill = method)) +
  geom_boxplot(na.rm = TRUE, alpha = 0.8, outlier.shape = NA) +
  facet_grid(rows = vars(livello_rumore),cols = vars(p, d),labeller = label_both)+
  scale_fill_manual(values = method_colors) +
  scale_y_log10() +
  labs(y = "Errore Post-Rotazione (Scala Log)",x = "Numero di Osservazioni (n)")+
  theme(legend.position = "none")

print(plot_procrustes_error_linear)

# --- GRAFICO NON-LINEARE ---
plot_procrustes_error_nonlinear <- all_results %>%
  filter(transformation == "nonlinear") %>%
  # MODIFICA: Anche qui, usiamo 'recon_err_rotated' per tutti i metodi
  ggplot(aes(x = as.factor(n), y = recon_err_rotated, fill = method)) +
  geom_boxplot(na.rm = TRUE, alpha = 0.8, outlier.shape = NA) +
  facet_grid( rows = vars(livello_rumore), cols = vars(p, d), labeller =  label_both)+
  scale_fill_manual(values = method_colors) +
  scale_y_log10() +
  labs(y = "Errore Post-Rotazione (Scala Log)",x = "Numero di Osservazioni (n)")+
  theme(legend.position = "none")

print(plot_procrustes_error_nonlinear)

cat("Grafici per l'errore post-rotazione (Procruste) creati con successo.\n")



# -------------------------------------------------------------------------
# Errore di ricostruzione dopo rotazione di procustes 
# standardizzando gli scores ottenuti dai vari metodi
# lavoriamo su all_results_augmented 

calculate_scaled_procrustes_error <- function(Z_true, Z_estimated) {
  if (any(is.na(Z_estimated))) return(NA)
  
  # --- MODIFICA CHIAVE: Standardizza entrambi i set di dati ---
  Z_estimated_scaled <- scale(Z_estimated)
  
  # Calcolo di Procruste (codice invariato)
  svd_results <- svd(t(Z_estimated_scaled) %*% Z_true)
  R <- svd_results$v %*% t(svd_results$u)
  Z_rotated <- Z_estimated_scaled %*% R
  
  mse <- mean(((Z_true - Z_rotated)%*%solve(cov(Z_true)))^2)
  return(mse)
}
# calcolo a posteriori dell'errore di ricostruzione
#   dello spazio latente dopo aver standardizzato gli score
all_results_augmented <- map_dfr(file_list, function(file_path) {
  env <- new.env()
  load(file_path, envir = env)
  
  # Estrai i dati dalla lista salvata
  run_output <- env$run_output
  results_tibble <- run_output$results
  latent_scores <- run_output$scores
  
  # Dobbiamo rigenerare Z_true con lo stesso seed per poter fare il calcolo.
  # Estraggo i parametri dal tibble.
  params <- results_tibble[1, c("n", "d", "p", "rep_id", "noise_level", "transformation")]
  set.seed(params$n + params$d*10 + params$p*100 + params$rep_id*1000 +
             ifelse(params$noise_level=="low", 0, 5000) +
             ifelse(params$transformation=="linear", 0, 10000))
  Z_true <- matrix(rnorm(params$n * params$d), nrow = params$n, ncol = params$d)
  
  # Calcola il nuovo errore per ogni metodo
  err_proc_scaled_pca <- calculate_scaled_procrustes_error(Z_true, latent_scores$pca)
  err_proc_scaled_fa  <- calculate_scaled_procrustes_error(Z_true, latent_scores$fa)
  err_proc_scaled_ae  <- calculate_scaled_procrustes_error(Z_true, latent_scores$ae)
  
  # Aggiungi la nuova metrica come colonna al tibble
  results_tibble <- results_tibble %>%
    mutate(
      recon_err_procrustes_scaled = case_when(
        method == "PCA" ~ err_proc_scaled_pca,
        method == "FA"  ~ err_proc_scaled_fa,
        method == "AE"  ~ err_proc_scaled_ae,
        TRUE ~ NA_real_
      )
    )
  
  return(results_tibble)
}, .progress = TRUE)

cat("Calcolo completato.\n")
glimpse(all_results_augmented)


all_results_augmented<- all_results_augmented %>%
  mutate(
    # Usa case_when() per tradurre i valori in una nuova colonna
    livello_rumore = case_when(
      noise_level == "low"  ~ "Basso",
      noise_level == "high" ~ "Alto",
      TRUE                  ~ noise_level # Lascia invariati altri valori se ce ne fossero
    ),
    # Converte la nuova colonna in un fattore per garantire l'ordine corretto ("Basso" prima di "Alto")
    livello_rumore = factor(livello_rumore, levels = c("Basso", "Alto"))
  )

plot_procrustes_error_linear_std <- all_results_augmented %>%
  filter(transformation == "linear") %>%
  # MODIFICA: Ora usiamo 'recon_err_rotated' per tutti i metodi sull'asse y
  ggplot(aes(x = as.factor(n), y =recon_err_procrustes_scaled, fill = method)) +
  geom_boxplot(na.rm = TRUE, alpha = 0.8, outlier.shape = NA) +
  facet_grid(rows = vars(livello_rumore),cols = vars(p, d),labeller = label_both)+
  scale_fill_manual(values = method_colors) +
  labs(y = "Errore Post-Rotazione (Scala Log)",x = "Numero di Osservazioni (n)")+
  theme(legend.position = "none")


# --- GRAFICO NON-LINEARE ---
plot_procrustes_error_nonlinear_std <- all_results_augmented %>%
  filter(transformation == "nonlinear") %>%
  # MODIFICA: Anche qui, usiamo 'recon_err_rotated' per tutti i metodi
  ggplot(aes(x = as.factor(n), y = recon_err_procrustes_scaled, fill = method)) +
  geom_boxplot(na.rm = TRUE, alpha = 0.8, outlier.shape = NA) +
  facet_grid( rows = vars(livello_rumore), cols = vars(p, d), labeller =  label_both)+
  scale_fill_manual(values = method_colors) +
  labs(y = "Errore Post-Rotazione (Scala Log)",x = "Numero di Osservazioni (n)")+
  theme(legend.position = "none")

print(plot_procrustes_error_linear_std)
print(plot_procrustes_error_nonlinear_std)
#nel caso lineare 

##################################
# Procustes relativizzato

# Questa funzione calcola l'errore di Procruste e lo normalizza (NMSE)
calculate_nmse_procrustes <- function(Z_true, Z_estimated) {
  # Se il modello è fallito, restituisci NA
  if (any(is.na(Z_estimated))) {
    return(NA)
  }
  
  # Calcola la varianza del segnale vero per la normalizzazione
  var_z_true <- mean(Z_true^2)
  # Aggiungi un controllo per il caso limite in cui Z_true sia tutto zero
  if (var_z_true < 1e-9) {
    return(NA) 
  }
  
  # Risolvi il problema di Procruste per trovare la rotazione ottimale R
  svd_results <- svd(t(Z_estimated) %*% Z_true)
  R <- svd_results$v %*% t(svd_results$u)
  
  # Applica la rotazione
  Z_rotated <- Z_estimated %*% R
  
  # Calcola l'errore assoluto (MSE)
  mse_absolute <- mean(((Z_true - Z_rotated)/Z_true)^2)
  
  # Calcola l'errore relativo (NMSE)
  nmse <- mse_absolute 
  
  return(nmse)
}


# --- 3. PROCESSO DI RICALCOLO ---

# Definisci la directory dove si trovano i risultati salvati
output_dir <- "risultati_simulazione3" # Adatta questo nome alla tua cartella

# Ottieni la lista di tutti i file .RData
file_list <- list.files(path = output_dir, pattern = "\\.RData$", full.names = TRUE)

if (length(file_list) == 0) {
  stop("Nessun file .RData trovato. Controlla il percorso in 'output_dir'.")
}

message(sprintf("Trovati %d file di risultati. Inizio il ricalcolo delle metriche...", length(file_list)))

# Usa map_dfr per iterare su ogni file e combinare i risultati in un unico dataframe
new_metrics_df <- map_dfr(file_list, function(file_path) {
  
  # Carica l'oggetto 'run_output' dal file in un ambiente temporaneo
  env_temp <- new.env()
  load(file_path, envir = env_temp)
  run_output <- env_temp$run_output
  
  # Estrai i parametri di questa run dalla prima riga della tabella dei risultati
  params <- run_output$results[1, ]
  
  # --- Ricostruisci il seme ESATTO usato per generare Z_true ---
  # NOTA: Questa formula deve corrispondere a quella usata nella tua simulazione originale!
  seed_value <- params$n + params$d*10 + params$p*100 + params$rep_id*1000 + 
                ifelse(params$noise_level == "low", 0, 5000) + 
                ifelse(params$transformation == "linear", 0, 10000)
  
  # Imposta il seme e rigenera la matrice Z_true
  set.seed(seed_value)
  Z_true <- matrix(rnorm(params$n * params$d), nrow = params$n, ncol = params$d)
  
  # Estrai gli scores stimati
  Z_pca <- run_output$scores$pca
  Z_fa <- run_output$scores$fa
  Z_ae <- run_output$scores$ae
  
  # Calcola la nuova metrica per ciascun metodo
  nmse_pca <- calculate_nmse_procrustes(Z_true, Z_pca)
  nmse_fa <- calculate_nmse_procrustes(Z_true, Z_fa)
  nmse_ae <- calculate_nmse_procrustes(Z_true, Z_ae)
  
  # Restituisci una tibble/dataframe con i parametri e le nuove metriche
  tibble(
    # Parametri identificativi
    n = params$n, 
    d = params$d, 
    p = params$p, 
    noise_level = params$noise_level, 
    transformation = params$transformation, 
    rep_id = params$rep_id,
    
    # Nuove metriche calcolate
    nmse_pca = nmse_pca,
    nmse_fa = nmse_fa,
    nmse_ae = nmse_ae
  )
}, .progress = TRUE) # .progress = TRUE mostra una barra di avanzamento!

# Il 'new_metrics_df' è in formato "largo" (una colonna per metrica/metodo).
# Trasformiamolo in formato "lungo" per poterlo unire a 'all_results'.
new_metrics_long_df <- new_metrics_df %>%
  pivot_longer(
    cols = c(nmse_pca, nmse_fa, nmse_ae),
    names_to = "method",
    values_to = "recon_err_rotated_relative",
    names_prefix = "nmse_"
  ) %>%
  mutate(method = toupper(method)) # Converte pca -> PCA, fa -> FA, etc.

# Unisci il dataframe originale con le nuove metriche
# Assumiamo che il tuo dataframe originale si chiami 'all_results'
all_results_updated <- left_join(
  all_results, 
  new_metrics_long_df,
  by = c("n", "d", "p", "noise_level", "transformation", "rep_id", "method")
)

# Per coerenza, creiamo anche la colonna "livello_rumore" in italiano
all_results_updated <- all_results_updated %>%
  mutate(
    livello_rumore = factor(noise_level, 
                            levels = c("low", "high"), 
                            labels = c("Basso", "Alto"))
  )


# --- NUOVO GRAFICO 1: Errore Relativo Post-Rotazione (NMSE) - SCENARI LINEARI ---
cat("\n--- Creazione Grafico Errore Relativo Post-Rotazione (Lineare) ---\n")

plot_procrustes_relative_error_linear <- all_results_updated %>%
  filter(transformation == "linear") %>%
  
  # --- MODIFICA CHIAVE: Usa la nuova colonna sull'asse y ---
  ggplot(aes(x = as.factor(n), y = recon_err_rotated_relative, fill = method)) +
  # -----------------------------------------------------------

geom_boxplot(na.rm = TRUE, alpha = 0.8, outlier.shape = NA) +
  facet_grid(
    rows = vars(livello_rumore), 
    cols = vars(p, d), 
    labeller = label_both
  ) +
  scale_fill_manual(values = method_colors) +
  
  # La scala logaritmica è ancora un'ottima scelta per visualizzare errori
  scale_y_log10() + 
  
  # --- MODIFICA CHIAVE: Aggiorna le etichette ---
  labs(
    y = "Errore Relativo Post-Rotazione (NMSE, Scala Log)",
    x = "Numero di Osservazioni (n)"
  ) +
  # ---------------------------------------------

theme_bw() + # Usiamo theme_bw() per coerenza visiva
  theme(legend.position = "none")

# Mostra il grafico
print(plot_procrustes_relative_error_linear)


# --- NUOVO GRAFICO 2: Errore Relativo Post-Rotazione (NMSE) - SCENARI NON-LINEARI ---
cat("\n--- Creazione Grafico Errore Relativo Post-Rotazione (Non-Lineare) ---\n")

plot_procrustes_relative_error_nonlinear <- all_results_updated %>%
  filter(transformation == "nonlinear") %>%
  
  # --- MODIFICA CHIAVE: Usa la nuova colonna sull'asse y ---
  ggplot(aes(x = as.factor(n), y = recon_err_rotated_relative, fill = method)) +
  # -----------------------------------------------------------

geom_boxplot(na.rm = TRUE, alpha = 0.8, outlier.shape = NA) +
  facet_grid(
    rows = vars(livello_rumore), 
    cols = vars(p, d), 
    labeller = label_both
  ) +
  scale_fill_manual(values = method_colors) +
  scale_y_log10() + 
  
  # --- MODIFICA CHIAVE: Aggiorna le etichette ---
  labs(
    y = "Errore Relativo Post-Rotazione (NMSE, Scala Log)",
    x = "Numero di Osservazioni (n)"
  ) +
  # ---------------------------------------------

theme_bw() +
  theme(legend.position = "none")

# Mostra il secondo grafico
print(plot_procrustes_relative_error_nonlinear)

# GRAFICO 3: Procustes scaled------------------------------------

# --- GRAFICO LINEARE ---
plot_scaled_procrustes_linear <- all_results_augmented %>%
  filter(transformation == "linear") %>%
  ggplot(aes(x = as.factor(n), y = recon_err_procrustes_scaled, fill = method)) +
  geom_boxplot(na.rm = TRUE, alpha = 0.8, outlier.shape = NA) +
  facet_grid(
    rows = vars(noise_level), cols = vars(p, d),
    labeller = label_both
  ) +
  scale_fill_manual(values = method_colors) +
  scale_y_log10() +
  labs(
    title = "Errore Geometrico Standardizzato (Procruste) - SCENARIO LINEARE",
    subtitle = "Misura la somiglianza della forma, eliminando l'effetto della scala",
    y = "Errore Post-Rotazione Standardizzato (Scala Log)",
    x = "Numero di Osservazioni (n)"
  )

print(plot_scaled_procrustes_linear)

# --- GRAFICO NON-LINEARE ---
plot_scaled_procrustes_nonlinear <- all_results_augmented %>%
  filter(transformation == "nonlinear") %>%
  ggplot(aes(x = as.factor(n), y = recon_err_procrustes_scaled, fill = method)) +
  geom_boxplot(na.rm = TRUE, alpha = 0.8, outlier.shape = NA) +
  facet_grid(
    rows = vars(noise_level), cols = vars(p, d),
    labeller = label_both
  ) +
  scale_fill_manual(values = method_colors) +
  scale_y_log10() +
  labs(
    title = "Errore Geometrico Standardizzato (Procruste) - SCENARIO NON-LINEARE",
    subtitle = "Misura la somiglianza della forma, eliminando l'effetto della scala",
    y = "Errore Post-Rotazione Standardizzato (Scala Log)",
    x = "Numero di Osservazioni (n)"
  )

print(plot_scaled_procrustes_nonlinear)








# ===================================================================
# SCRIPT UNICO PER ANALISI SCREE PLOT (LINEARE E NON-LINEARE)
# ===================================================================

# --- Assicurati di avere le librerie e le funzioni di supporto caricate ---
# library(keras); library(tensorflow); library(dplyr); library(purrr);
# library(ggplot2); library(tidyr)
# E le funzioni: generate_sparse_weights_controlled, callback_list

# --- 1. FUNZIONE PER L'ANALISI SCREE PLOT (invariata, usa factanal) ---

# Funzione completa e verificata
run_scree_analysis <- function(X_scaled, d_max, p, callbacks) {
  
  results_list <- list()
  
  # 1. PCA (invariato)
  pca_model <- prcomp(X_scaled, center = FALSE, scale. = FALSE)
  for (d_est in 1:d_max) {
    if (d_est <= ncol(pca_model$x)) {
      X_recon_pca <- pca_model$x[, 1:d_est, drop = FALSE] %*% t(pca_model$rotation[, 1:d_est, drop = FALSE])
      mse_pca <- mean((X_scaled - X_recon_pca)^2)
      results_list[[length(results_list) + 1]] <- tibble(d_est = d_est, method = "PCA", mse = mse_pca)
    }
  }
  
  # 2. Analisi Fattoriale (FA) - AGGIORNATA E ROBUSTA
  for (d_est in 1:d_max) {
    df_fa <- ((p - d_est)^2 - p - d_est) / 2
    
    if (df_fa < 0) {
      mse_fa <- NA # Impossibile matematicamente
    } else {
      fa_model <- tryCatch({
        factanal(X_scaled, factors = d_est, rotation = "varimax", scores = "regression")
      }, error = function(e) { NULL })
      
      mse_fa <- if (!is.null(fa_model) && !is.null(fa_model$scores)) {
        X_recon_fa <- fa_model$scores %*% t(fa_model$loadings)
        mean((X_scaled - X_recon_fa)^2)
      } else { NA } # Fallimento per altre ragioni (es. convergenza)
    }
    results_list[[length(results_list) + 1]] <- tibble(d_est = d_est, method = "FA", mse = mse_fa)
  }
  
  # 3. Autoencoder (AE) - IL TUO CODICE ORIGINALE, CHE È GIÀ CORRETTO E FLESSIBILE
  for (d_est in 1:d_max) {
    k_clear_session()
    units_l1 <- min(512, round(p * 1.5)); units_l2 <- max(d_est + 5, round((units_l1 + d_est) / 2))
    ae_model <- keras_model_sequential() %>%
      layer_dense(units = units_l1, activation = "tanh", input_shape = p, kernel_initializer = initializer_glorot_normal()) %>%
      layer_dense(units = units_l2, activation = "tanh", kernel_initializer = initializer_glorot_normal()) %>%
      layer_dense(units = d_est, name = "code", activation = "linear") %>%
      layer_dense(units = units_l2, activation = "tanh", kernel_initializer = initializer_glorot_normal()) %>%
      layer_dense(units = units_l1, activation = "tanh", kernel_initializer = initializer_glorot_normal()) %>%
      layer_dense(units = p, activation = "linear")
    
    ae_model %>% compile(optimizer = "adam", loss = "mse")
    
    # Eseguire il fit solo se ci sono dati
    if (nrow(X_scaled) > 0) {
      ae_model %>% fit(X_scaled, X_scaled, epochs = 500, batch_size = 64, callbacks = callbacks, validation_split = 0.2, verbose = 0)
      X_recon_ae <- predict(ae_model, X_scaled)
      mse_ae <- mean((X_scaled - X_recon_ae)^2)
    } else {
      mse_ae <- NA # Gestione di un caso limite
    }
    
    results_list[[length(results_list) + 1]] <- tibble(d_est = d_est, method = "AE", mse = mse_ae)
  }
  
  return(bind_rows(results_list))
}

# --- 2. SCRIPT DI ORCHESTRAZIONE UNIFICATO ---

n_replications_scree <- 25
output_dir_scree <- "screeplotresults"
if (!dir.exists(output_dir_scree)) { dir.create(output_dir_scree) }

# Scenari originali (con d_true esplicito)
grid1 <- expand.grid(
  n = c(1000, 10000),
  noise_level = c("low", "high"),
  transformation = c("linear", "nonlinear"),
  p = 10,
  d_true = 2,
  stringsAsFactors = FALSE
)

# Nuovi scenari: p=10, d_true=5
grid2 <- expand.grid(
  n = c(1000, 10000),
  noise_level = c("low", "high"),
  transformation = c("linear", "nonlinear"),
  p = 10,
  d_true = 5,
  stringsAsFactors = FALSE
)

# Nuovi scenari: p=100, d_true=2
grid3 <- expand.grid(
  n = c(1000, 10000),
  noise_level = c("low", "high"),
  transformation = c("linear", "nonlinear"),
  p = 100,
  d_true = 2,
  stringsAsFactors = FALSE
)

# Nuovi scenari: p=100, d_true=5
grid4 <- expand.grid(
  n = c(1000, 10000),
  noise_level = c("low", "high"),
  transformation = c("linear", "nonlinear"),
  p = 100,
  d_true = 5,
  stringsAsFactors = FALSE
)
grid5 <- expand.grid(
  n = c(1000, 10000),
  noise_level = c("low", "high"),
  transformation = c("linear", "nonlinear"),
  p = 1000,
  d_true = 2,
  stringsAsFactors = FALSE
)
grid6 <- expand.grid(
  n = c(1000, 10000),
  noise_level = c("low", "high"),
  transformation = c("linear", "nonlinear"),
  p = 1000,
  d_true = 5,
  stringsAsFactors = FALSE
)


params_scree_grid <- bind_rows(grid5, grid6)


all_scree_results_list <- list()


#-----------------------------------------------------------------------
# Assicurati che questi oggetti/funzioni siano definiti prima di eseguire il ciclo:
#
# 1. params_scree_grid: Il dataframe con tutti gli scenari da testare.
# 2. n_replications_scree: Il numero di repliche per ogni scenario (es. 50).
# 3. output_dir_scree: La cartella dove salvare i risultati (es. "screeplotresults_esteso").
# 4. run_scree_analysis(): La funzione che esegue PCA, FA, AE per un d_max.
# 5. generate_sparse_weights_controlled(): La tua funzione per creare la matrice W.
# 6. callback_list: La lista di callback per Keras (es. callback_early_stopping).
#-----------------------------------------------------------------------

# Ciclo principale che itera su ogni riga della griglia dei parametri (ogni scenario)
for (i in 1:nrow(params_scree_grid)) {
  
  # Estrai i parametri per lo scenario corrente
  current_params <- params_scree_grid[i, ]
  
  # --- GESTIONE DINAMICA DI d_max ---
  # Imposta il numero massimo di dimensioni da testare in base alla d_true dello scenario
  d_max <- ifelse(current_params$d_true == 2, 4, 7)
  
  # Messaggio di log per tenere traccia del progresso
  message(sprintf("--- INIZIO SCENARIO %d/%d (p=%d, d=%d, n=%d, noise=%s, trans=%s) ---", 
                  i, nrow(params_scree_grid), current_params$p, current_params$d_true,
                  current_params$n, current_params$noise_level, current_params$transformation))
  
  # --- Generazione della matrice dei pesi W (una volta per scenario) ---
  # Usa i parametri dello scenario corrente per generare W
  set.seed(current_params$p * 10 + current_params$d_true) # Seed riproducibile per W
  W_sparse <- generate_sparse_weights_controlled(d = current_params$d_true, p = current_params$p)
  
  # Adatta la struttura di W in base al tipo di trasformazione (lineare o non-lineare)
  if (current_params$transformation == "linear") {
    W <- W_sparse
  } else {
    # Per il non-lineare, dividi W in due parti per le due diverse trasformazioni
    p_half <- current_params$p / 2
    W <- list(part1 = W_sparse[, 1:p_half], part2 = W_sparse[, (p_half + 1):current_params$p])
  }
  
  # --- Ciclo interno sulle replicazioni per lo scenario corrente ---
  for (rep_id in 21:n_replications_scree) {
    message(sprintf("    -> Avvio replica %d/%d...", rep_id, n_replications_scree))
    
    # --- A. Generazione Dati (per ogni replica) ---
    set.seed(i * 1000 + rep_id) # Seed riproducibile per i dati di ogni replica
    
    # 1. Genera le variabili latenti Z
    Z <- matrix(rnorm(current_params$n * current_params$d_true), 
                nrow = current_params$n, ncol = current_params$d_true)
    
    # 2. Applica le trasformazioni per generare il segnale X
    if (current_params$transformation == "linear") {
      X <- Z %*% W
    } else {
      p_half <- current_params$p / 2
      X <- matrix(0, nrow = current_params$n, ncol = current_params$p)
      
      Z_part1_transformed <- Z^3
      Z_part2_transformed <- exp(0.5*Z)
      
      X[, 1:p_half] <- scale(Z_part1_transformed) %*% W$part1
      X[, (p_half + 1):current_params$p] <- scale(Z_part2_transformed) %*% W$part2
    }
    
    # 3. Aggiungi rumore
    noise_sd_prop <- ifelse(current_params$noise_level == "low", 0.1, 0.5)
    noise_sd_val <- noise_sd_prop * sd(as.vector(X))
    noise <- matrix(rnorm(current_params$n * current_params$p, mean = 0, sd = noise_sd_val), 
                    nrow = current_params$n, ncol = current_params$p)
    X_final <- X + noise
    
    # 4. Scala i dati finali (input per i modelli)
    X_scaled <- scale(X_final)
    
    # --- B. Esecuzione dell'analisi per la replica corrente ---
    # Chiama la funzione che calcola gli MSE per d_est da 1 a d_max
    results_from_run <- run_scree_analysis(
      X_scaled = X_scaled, 
      d_max = d_max, 
      p = current_params$p, 
      callbacks = callback_list
    )
    
    # --- C. Aggiunta dei metadati e salvataggio ---
    # Aggiungi le colonne con i parametri dello scenario e della replica al dataframe dei risultati
    single_run_results <- results_from_run %>%
      mutate(
        n = current_params$n,
        noise_level = current_params$noise_level,
        transformation = current_params$transformation,
        p = current_params$p,
        d_true = current_params$d_true,
        rep_id = rep_id
      )
    
    # Crea un nome di file descrittivo che include tutti i parametri chiave
    file_name <- sprintf("%s/scree_p%d_d%d_n%d_noise%s_trans%s_rep%d.RData",
                         output_dir_scree, 
                         current_params$p, current_params$d_true, current_params$n, 
                         current_params$noise_level, substr(current_params$transformation, 1, 4), 
                         rep_id)
    
    # Salva il dataframe dei risultati di questa singola replica
    save(single_run_results, file = file_name)
    
  } # Fine del ciclo sulle replicazioni
  
} # Fine del ciclo sugli scenari

# 3. Ottieni la lista di tutti i file .RData salvati
# list.files cerca i file in una cartella.
# - path: la cartella in cui cercare.
# - pattern: cerca solo i file che finiscono con ".RData".
# - full.names = TRUE: restituisce il percorso completo, es. "screeplotresults/file1.RData"
#                      che è necessario per la funzione load().
file_list <- list.files(path = output_dir_scree, pattern = "\\.RData$", full.names = TRUE)

# Controllo opzionale: verifica di aver trovato i file
if (length(file_list) == 0) {
  stop("Nessun file .RData trovato nella directory '", output_dir_scree, "'. Controlla il percorso o la directory di lavoro.")
} else {
  message(sprintf("Trovati %d file .RData. Inizio il caricamento...", length(file_list)))
}


# 4. Ciclo per caricare i file e ricostruire la lista
all_scree_results_list <- list() # Inizializza la lista vuota, come hai perso

for (file_path in file_list) {
  # La funzione load() carica gli oggetti dal file nell'ambiente corrente.
  # Nel tuo caso, ogni file contiene un solo oggetto chiamato 'single_run_results'.
  load(file_path)
  
  # Aggiungi l'oggetto appena caricato ('single_run_results') alla tua lista
  all_scree_results_list[[length(all_scree_results_list) + 1]] <- single_run_results
}

# Messaggio di conferma
message("Caricamento completato. Tutti i risultati sono stati aggiunti alla lista.")

all_scree_results_df <- bind_rows(all_scree_results_list)

# --- Funzione di Plotting Generale ---
# Questa funzione crea una figura a 8 pannelli per uno specifico scenario (p, d_true)
create_scenario_plot <- function(df, p_val, d_val) {
  
  # 1. Filtra i dati per lo scenario specifico
  plot_data <- df %>% 
    filter(p == p_val, d_true == d_val)
  
  # 2. Determina dinamicamente i breaks per l'asse x
  d_max <- if (d_val == 2) 4 else 7
  x_breaks <- 1:d_max
  
  # 3. Crea il grafico con la nuova struttura di faccette
  plot_object <- ggplot(plot_data, aes(x = d_est, y = mse, color = method, fill = method)) +
    stat_summary(fun.data = mean_se, geom = "ribbon", alpha = 0.15, aes(color = NULL), na.rm = TRUE) +
    stat_summary(fun = mean, geom = "line", size = 1, na.rm = TRUE) +
    stat_summary(fun = mean, geom = "point", size = 2.5, na.rm = TRUE) +
    
    # --- LA NUOVA STRUTTURA A 8 PANNELLI ---
    # Le righe sono definite dalla combinazione di n e noise_level
    # Le colonne sono definite dalla trasformazione (lineare a sx, non-lineare a dx)
    facet_grid(rows = vars(n, noise_level), 
               cols = vars(transformation), 
               labeller = label_both) +
    
    scale_fill_manual(values = method_colors) +
    scale_color_manual(values = method_colors) +
    
    # --- ETICHETTE E TITOLI SEMPLIFICATI ---
    labs(
      title = sprintf("Errore di Ricostruzione per p = %d e d_vera = %d", p_val, d_val),
      x = "Dimensioni Latenti Stimate (d_est)",
      y = "Errore Quadratico Medio (MSE)"
    ) +
    
    scale_x_continuous(breaks = x_breaks) +
    theme_bw(base_size = 12) + # Riduciamo un po' la base_size per grafici più compatti
    
    # --- TEMA SEMPLIFICATO ---
    theme(
      strip.text = element_text(face = "bold", size = 9), 
      legend.position = "bottom",
      legend.title = element_blank() # Rimuove il titolo della legenda ("Metodo")
    )
  
  return(plot_object)
}

# --- Ciclo per Generare e Salvare Tutti i Grafici ---

# 1. Trova tutte le combinazioni uniche di (p, d_true) nei tuoi risultati
scenarios_to_plot <- all_scree_results_df %>%
  distinct(p, d_true) %>%
  arrange(p, d_true) # Ordina per coerenza

# 2. Crea una cartella per salvare le nuove figure
output_dir_plots <- "final_plots"
if (!dir.exists(output_dir_plots)) { dir.create(output_dir_plots) }

# 3. Itera su ogni scenario, crea il grafico e salvalo
for (i in 1:nrow(scenarios_to_plot)) {
  p_current <- scenarios_to_plot$p[i]
  d_current <- scenarios_to_plot$d_true[i]
  
  message(sprintf("Generazione grafico per p=%d, d_true=%d...", p_current, d_current))
  
  # Crea il grafico
  current_plot <- create_scenario_plot(all_scree_results_df, p_current, d_current)
  
  # Mostra il grafico nella sessione R
  print(current_plot)
  
  # Salva il grafico in un file PNG di alta qualità
  file_name <- sprintf("%s/plot_p%d_d%d.png", output_dir_plots, p_current, d_current)
  ggsave(
    filename = file_name,
    plot = current_plot,
    width = 8, 
    height = 10, 
    dpi = 300
  )
}



scores_summary_df <- map_dfr(file_list, function(file_path) {
  
  # Carica l'oggetto 'run_output' dal file in un ambiente temporaneo
  env_temp <- new.env()
  load(file_path, envir = env_temp)
  run_output <- env_temp$run_output
  
  # Estrai i parametri di questa run
  params <- run_output$results[1, ]
  
  # Estrai la lista degli scores
  scores <- run_output$scores
  
  # Itera su ciascun metodo (pca, fa, ae) per calcolare le statistiche
  map_dfr(names(scores), function(method_name) {
    
    Z_estimated <- scores[[method_name]]
    
    # Se gli scores non esistono o sono tutti NA (es. FA fallita), restituisci una riga di NA
    if (is.null(Z_estimated) || all(is.na(Z_estimated))) {
      tibble(
        n = params$n, d = params$d, p = params$p, 
        noise_level = params$noise_level, transformation = params$transformation, rep_id = params$rep_id,
        method = toupper(method_name),
        variance = NA_real_,
        min_score = NA_real_,
        max_score = NA_real_
      )
    } else {
      # Altrimenti, calcola le statistiche
      tibble(
        n = params$n, d = params$d, p = params$p, 
        noise_level = params$noise_level, transformation = params$transformation, rep_id = params$rep_id,
        method = toupper(method_name),
        # Varianza calcolata su tutti gli scores della matrice
        variance = var(as.vector(Z_estimated), na.rm = TRUE),
        min_score = min(Z_estimated, na.rm = TRUE),
        max_score = max(Z_estimated, na.rm = TRUE)
      )
    }
  })
}, .progress = TRUE) # Mostra una barra di avanzamento
print(head(scores_summary_df))
