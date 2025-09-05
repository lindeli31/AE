
#################################################################
# SIMULATION STUDY: PCA vs FA vs AE
# ---------------------------------------------------------------
# Compares Principal Component Analysis (PCA), Factor Analysis (FA),
# and a simple Autoencoder (AE) on synthetic data under linear and
# non-linear generative models, across multiple noise levels and sizes.
#
# OUTPUT
# - For each replication and parameter setting, saves an .RData file
#   containing:
#     run_output$results : tibble of metrics for PCA/FA/AE
#     run_output$scores  : list with latent scores (pca/fa/ae)
#
# REPRODUCIBILITY
# - Seeds are set per (n, d, p, replication, noise, transformation).
#
# REQUIREMENTS
# - R packages listed below (install line provided).
#################################################################

# --- 1) LIBRARIES & SETUP --------------------------------------
# Make sure these packages are installed:
# install.packages(c("MASS","psych","keras","dplyr","purrr","pracma","clue","ggplot2","tidyr","knitr","gtools","ks"))

library(MASS)        # mvrnorm (if needed)
library(psych)       # Factor Analysis (fa)
library(keras)
library(tensorflow)
library(dplyr)       # data manipulation (tibble)
library(purrr)       # functional helpers
library(pracma)      # Procrustes idea (we use SVD directly)
library(clue)        # solve_LSAP for optimal column matching
library(ggplot2)
library(tidyr)
library(knitr)
library(gtools)
library(ks)          # (unused here, kept to avoid changing dependencies)

# Early stopping & LR scheduling for AE training
callback_list <- list(
  callback_early_stopping(
    monitor = "val_loss",
    patience = 40,
    min_delta = 0.0001,      # minimum meaningful improvement
    mode = "min",
    restore_best_weights = TRUE
  ),
  callback_reduce_lr_on_plateau(
    monitor = "val_loss",
    factor = 0.5,
    patience = 20,
    min_delta = 0.001,       # plateau threshold
    min_lr = 1e-7,
    verbose = 10
  )
)

# --- 2) SUPPORT FUNCTIONS & METRICS ----------------------------

#' Calculate alignment-aware metrics between true and estimated latent factors.
#'
#' This routine (i) permutes and flips signs to best match columns,
#' (ii) reports factor-wise correlation (squared, averaged),
#' (iii) reconstruction MSE after best permutation/sign,
#' (iv) reconstruction MSE after an orthogonal Procrustes rotation (via SVD).
#'
#' @param Z_true      n x d matrix of ground-truth latent factors
#' @param Z_estimated n x d matrix of estimated factors (may contain NA)
#' @return list(factor_cor, recon_err_permuted, recon_err_rotated)
calculate_all_metrics <- function(Z_true, Z_estimated) {
  if (any(is.na(Z_estimated))) {
    return(list(factor_cor = NA, recon_err_permuted = NA, recon_err_rotated = NA))
  }
  
  # --- 1) BEST PERMUTATION & SIGN MATCHING ---------------------
  C <- abs(cor(Z_true, Z_estimated))
  perm <- clue::solve_LSAP(C, maximum = TRUE)     # assigns estimated cols to true cols
  Z_permuted <- Z_estimated[, perm]
  signs <- sign(diag(cor(Z_true, Z_permuted)))
  Z_aligned <- sweep(Z_permuted, 2, signs, FUN = "*")
  
  # --- 2) PERFORMANCE METRICS ----------------------------------
  # Metric A: average squared factor correlation (diagonal of corr matrix)
  factor_cor_val <- mean(diag(cor(Z_true, Z_aligned))^2)
  
  # Metric B: MSE after best permutation/sign
  recon_err_permuted_val <- mean((Z_true - Z_aligned)^2)
  
  # --- 3) ORTHOGONAL PROCRUSTES ROTATION -----------------------
  # Find orthogonal R that best maps Z_estimated to Z_true (in least squares sense).
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

#' Generate a sparse, column-wise controlled loading matrix W.
#'
#' Each observed variable j draws a Dirichlet allocation across d factors,
#' random signs, and a total magnitude sampled from a "strong" or "weak"
#' range, according to the requested proportion of strong indicators.
#'
#' @param d number of latent factors
#' @param p number of observed variables
#' @param alpha_val Dirichlet concentration for sparsity (smaller -> sparser)
#' @param strong_indicator_proportion share of variables considered "strong"
#' @param strong_magnitude_range magnitude range used for strong variables
#' @param weak_magnitude_range   magnitude range used for weak variables
#' @return d x p loading matrix W
generate_sparse_weights_controlled <- function(d, p,
                                               alpha_val = 0.1,
                                               strong_indicator_proportion = 0.75,
                                               strong_magnitude_range = c(3, 7),
                                               weak_magnitude_range = c(0.5, 1.5)) {
  
  # 1) Decide how many strong vs weak indicators
  num_strong <- floor(p * strong_indicator_proportion)
  num_weak <- p - num_strong
  
  # 2) Randomly assign indices to strong/weak roles
  shuffled_indices <- sample(1:p)
  strong_indices <- shuffled_indices[1:num_strong]
  weak_indices <- if (num_weak > 0) shuffled_indices[(num_strong + 1):p] else integer(0)
  
  # Initialize W
  W <- matrix(0, nrow = d, ncol = p)
  
  # 3) Build each column j
  for (j in 1:p) {
    # Dirichlet proportions across factors (encourages sparsity)
    proportions <- t(rdirichlet(1, rep(alpha_val, d)))
    signs <- sample(c(-1, 1), size = d, replace = TRUE)
    
    # Strong/weak total magnitude for variable j
    if (j %in% strong_indices) {
      total_magnitude <- runif(1, min = strong_magnitude_range[1], max = strong_magnitude_range[2])
    } else {
      total_magnitude <- runif(1, min = weak_magnitude_range[1], max = weak_magnitude_range[2])
    }
    
    # Final column
    W[, j] <- signs * proportions * total_magnitude
  }
  
  return(W)
}

# --- 3) MAIN SINGLE-SIMULATION ROUTINE -------------------------

#' Run one simulation instance for given (n, d, p, noise, transformation).
#'
#' Steps:
#'  A) Generate latent Z and observed X under linear or non-linear mapping.
#'  B) Fit PCA, FA, and an AE; extract latent codes.
#'  C) Compute alignment-aware metrics and data-space reconstruction errors.
#'
#' @param n sample size
#' @param d latent dimensionality
#' @param p observed dimensionality
#' @param noise_level "low" or "high"
#' @param transformation "linear" or "nonlinear"
#' @param replication_id integer id for seeding
#' @param W either a d x p matrix (linear) or a list with $part1,$part2 (nonlinear)
#' @return list(results = tibble, scores = list(pca, fa, ae))
run_single_simulation <- function(n, d, p, noise_level, transformation, replication_id, W) {
  
  # --- A) DATA GENERATION --------------------------------------
  set.seed(n + d*10 + p*100 + replication_id*1000 +
             ifelse(noise_level == "low", 0, 5000) +
             ifelse(transformation == "linear", 0, 10000))
  tf$random$set_seed(n + d*10 + p*100 + replication_id*1000)
  
  Z <- matrix(rnorm(n * d), nrow = n, ncol = d)
  X <- matrix(0, nrow = n, ncol = p)
  
  if (transformation == "linear") {
    X <- Z %*% W
  } else {
    # Non-linear blocks: Z^3 and exp(0.5*Z), each standardized before mixing
    p_part1 <- p / 2
    
    Z_part1_transformed <- Z^3
    Z_part2_transformed <- exp(0.5 * Z)
    
    Z_part1_std <- scale(Z_part1_transformed)
    Z_part2_std <- scale(Z_part2_transformed)
    
    X[, 1:p_part1]        <- Z_part1_std %*% W$part1
    X[, (p_part1 + 1):p]  <- Z_part2_std %*% W$part2
  }
  
  noise_sd_val <- ifelse(noise_level == "low", 0.1, 0.5)
  noise <- matrix(
    rnorm(n * p, mean = 0, sd = noise_sd_val * sd(as.vector(X))),
    nrow = n, ncol = p
  )
  
  X_final  <- X + noise
  X_scaled <- scale(X_final)
  
  # --- B) MODELS ------------------------------------------------
  # PCA
  pca_model <- prcomp(X_scaled, center = FALSE, scale. = FALSE)
  message("PCA completed")
  Z_pca <- pca_model$x[, 1:d]
  
  # FA (may fail for some settings; handled via tryCatch)
  fa_model <- tryCatch({
    psych::fa(X_scaled, nfactors = d, rotate = "varimax", fm = "ml", scores = "regression")
  }, error = function(e) { NULL })
  message("FA completed")
  Z_fa <- if (!is.null(fa_model)) fa_model$scores else matrix(NA, nrow = n, ncol = d)
  
  # AE (symmetric, tanh activations, linear code)
  k_clear_session()
  
  units_l1 <- min(1024, round(p * 1.2))
  compression_ratio <- (d / units_l1)^(1/3)
  units_l2_ideal <- round(units_l1 * compression_ratio)
  units_l3_ideal <- round(units_l2_ideal * compression_ratio)
  units_l3 <- max(units_l3_ideal, d + 1)
  units_l2 <- max(units_l2_ideal, units_l3 + 1)
  if (units_l1 <= units_l2) { units_l1 <- units_l2 + 1 }
  
  ae_model <- keras_model_sequential(name = "Autoencoder") %>%
    layer_dense(units = units_l1, activation = "tanh",
                input_shape = ncol(X_scaled),
                kernel_initializer = initializer_glorot_normal()) %>%
    layer_dense(units = units_l2, activation = "tanh",
                kernel_initializer = initializer_glorot_normal()) %>%
    layer_dense(units = units_l3, activation = "tanh",
                kernel_initializer = initializer_glorot_normal()) %>%
    layer_dense(units = d, name = "code", activation = "linear") %>%
    layer_dense(units = units_l3, activation = "tanh",
                kernel_initializer = initializer_glorot_normal()) %>%
    layer_dense(units = units_l2, activation = "tanh",
                kernel_initializer = initializer_glorot_normal()) %>%
    layer_dense(units = units_l1, activation = "tanh",
                kernel_initializer = initializer_glorot_normal()) %>%
    layer_dense(units = ncol(X_scaled), activation = "linear")
  
  ae_model %>% compile(optimizer = "adam", loss = "mse")
  
  # IMPORTANT: fit() returns the training history (we store it for val_loss)
  history <- ae_model %>% fit(
    X_scaled, X_scaled,
    epochs = 1000,
    batch_size = 64,
    callbacks = callback_list,
    validation_split = 0.2,
    verbose = 0
  )
  
  encoder_model <- keras_model(
    inputs = ae_model$input,
    outputs = get_layer(ae_model, "code")$output
  )
  Z_ae <- predict(encoder_model, X_scaled)
  message("AE completed")
  
  # --- C) METRICS -----------------------------------------------
  metrics_pca <- calculate_all_metrics(Z, Z_pca)
  metrics_fa  <- calculate_all_metrics(Z, Z_fa)
  metrics_ae  <- calculate_all_metrics(Z, Z_ae)
  message("Metrics computed")
  
  # Data-space reconstruction losses (MSE)
  # AE: best validation MSE during training
  data_recon_mse_ae <- suppressWarnings(min(history$metrics$val_loss, na.rm = TRUE))
  
  # PCA: project back using top-d components
  X_recon_pca <- Z_pca %*% t(pca_model$rotation[, 1:d])
  data_recon_mse_pca <- mean((X_scaled - X_recon_pca)^2)
  
  # FA: regression scores times loadings
  data_recon_mse_fa <- if (!is.null(fa_model)) {
    X_recon_fa <- Z_fa %*% t(fa_model$loadings)
    mean((X_scaled - X_recon_fa)^2)
  } else {
    NA
  }
  
  # Consolidate results
  results_tibble <- tibble(
    n = n, d = d, p = p,
    noise_level = noise_level,
    transformation = transformation,
    rep_id = replication_id,
    method = c("PCA", "FA", "AE"),
    factor_cor = c(metrics_pca$factor_cor,  metrics_fa$factor_cor,  metrics_ae$factor_cor),
    recon_err_permuted = c(metrics_pca$recon_err_permuted, metrics_fa$recon_err_permuted, metrics_ae$recon_err_permuted),
    recon_err_rotated  = c(metrics_pca$recon_err_rotated,  metrics_fa$recon_err_rotated,  metrics_ae$recon_err_rotated),
    data_recon_mse     = c(data_recon_mse_pca, data_recon_mse_fa, data_recon_mse_ae)
  )
  
  list(
    results = results_tibble,
    scores  = list(pca = Z_pca, fa = Z_fa, ae = Z_ae)
  )
}

# --- 4) ORCHESTRATION SCRIPT -----------------------------------

n_replications <- 50
output_dir <- "risultati_simulazione3"

# ensure output directory exists (small convenience)
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

params_unique <- expand.grid(
  n = c(1000, 10000),
  d = c(2, 5),
  p = c(10, 100, 1000),
  noise_level = c("low", "high"),
  transformation = c("linear", "nonlinear"),
  stringsAsFactors = FALSE
)

# for (i in 1:nrow(params_unique)) {
#   current_params <- params_unique[i, ]
#   
#   message(sprintf(
#     "--- START PARAM SET %d/%d: n=%d, d=%d, p=%d, noise=%s, transf=%s ---",
#     i, nrow(params_unique),
#     current_params$n, current_params$d, current_params$p,
#     current_params$noise_level, current_params$transformation
#   ))
#   
#   # Seed for weight generation consistency across replications
#   set.seed(
#     current_params$n + current_params$d*10 + current_params$p*100 +
#       ifelse(current_params$transformation == "linear", 0, 10000)
#   )
#   
#   # 1) Generate sparse/controlled loading matrix
#   W_sparse <- generate_sparse_weights_controlled(
#     d = current_params$d,
#     p = current_params$p
#   )
#   
#   # 2) Build W for linear / nonlinear scenarios
#   if (current_params$transformation == "linear") {
#     W <- W_sparse
#   } else {
#     p_part1 <- current_params$p / 2
#     W <- list(
#       part1 = W_sparse[, 1:p_part1],
#       part2 = W_sparse[, (p_part1 + 1):current_params$p]
#     )
#   }
#   
#   # Replications
#   for (rep_id in 1:n_replications) {
#     message(sprintf("  -> Replication %d/%d ...", rep_id, n_replications))
#     
#     run_output <- run_single_simulation(
#       n = current_params$n,
#       d = current_params$d,
#       p = current_params$p,
#       noise_level = current_params$noise_level,
#       transformation = current_params$transformation,
#       replication_id = rep_id,
#       W = W
#     )
#     
#     file_name <- sprintf(
#       "%s/res_n%d_d%d_p%d_%s_%s_rep%d.RData",
#       output_dir,
#       current_params$n, current_params$d, current_params$p,
#       current_params$noise_level, current_params$transformation, rep_id
#     )
#     
#     # Save full run_output object (results + scores)
#     save(run_output, file = file_name)
#   }
#   
#   message(sprintf("--- END PARAM SET %d/%d ---", i, nrow(params_unique)))
# }


# analysis -----------------------------------------------------------------



# --- 1. CARICAMENTO DATI ---

output_dir <- "risultati_simulazione" 

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



# -------------------------------------------------------------------------
# Errore di ricostruzione dopo rotazione di procustes 
# standardizzando gli scores ottenuti dai vari metodi
# lavoriamo su all_results_augmented 

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

plot_procrustes_error_linear_std <- all_results_augmented %>%
  filter(transformation == "linear") %>%
  # MODIFICA: Ora usiamo 'recon_err_rotated' per tutti i metodi sull'asse y
  ggplot(aes(x = as.factor(n), y =recon_err_procrustes_scaled, fill = method)) +
  geom_boxplot(na.rm = TRUE, alpha = 0.8, outlier.shape = NA) +
  facet_grid(rows = vars(livello_rumore),cols = vars(p, d),labeller = label_both)+
  scale_fill_manual(values = method_colors) +
  scale_y_log10()+
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
  scale_y_log10()+
  labs(y = "Errore Post-Rotazione (Scala Log)",x = "Numero di Osservazioni (n)")+
  theme(legend.position = "none")

print(plot_procrustes_error_linear_std)
print(plot_procrustes_error_nonlinear_std)
#nel caso lineare 



