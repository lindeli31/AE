# PCA vs FA vs AE — Simulation Study (R)

This repository contains the R code for a small simulation study used in my BSc thesis. The script compares **Principal Component Analysis (PCA)**, **Factor Analysis (FA)**, and a simple **Autoencoder (AE)** on synthetic data generated under **linear** and **non-linear** mappings, at different noise levels and dimensions.

> **TL;DR:** run `simulazione.R` — it generates data, fits PCA/FA/AE, computes alignment-aware metrics, and saves results for each replication/setting.

---

## What the script does

- **Data generation:** draws latent variables \(Z\) and produces observed data \(X\) via a sparse loading matrix.  
  - *Linear:* \(X = Z W\)  
  - *Non-linear:* blocks based on \(Z^3\) and \(\exp(0.5 Z)\), standardized before mixing.
- **Models:**  
  - PCA (top-\(d\) components)  
  - FA (ML, varimax; scores by regression)  
  - AE (symmetric MLP with `tanh`, linear code layer, early stopping & ReduceLROnPlateau)
- **Metrics (alignment-aware):**  
  - Average **squared factor correlation** after best permutation/sign  
  - **MSE** after best permutation/sign  
  - **MSE** after **orthogonal Procrustes** rotation  
  - **Data-space MSE** (PCA/FA reconstruction; AE uses best validation loss)
- **Grid & replications:** loops over \(n \in \{1000, 10000\}\), \(d \in \{2,5\}\), \(p \in \{10,100,1000\}\), noise \(\in\) {low, high}, transformation \(\in\) {linear, nonlinear}, with `n_replications = 50`.

---

## File layout

- `simulazione.R` — the entire study (single file)
- Results are written to `risultati_simulazione3/` as `.RData` files (created automatically).

---

## Requirements

- R packages: `MASS`, `psych`, `keras`, `tensorflow`, `dplyr`, `purrr`, `pracma`, `clue`, `ggplot2`, `tidyr`, `knitr`, `gtools`, `ks`  
  Install in R with:
  ```r
  install.packages(c("MASS","psych","keras","dplyr","purrr","pracma","clue","ggplot2","tidyr","knitr","gtools","ks"))
