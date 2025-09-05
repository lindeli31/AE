# PCA vs FA vs AE — Simulation Study (R)

This repository contains the R code for a small simulation study used in my BSc thesis. The script compares **Principal Component Analysis (PCA)**, **Factor Analysis (FA)**, and a simple **Autoencoder (AE)** on synthetic data generated under **linear** and **non-linear** mappings, at different noise levels and dimensions.

> **TL;DR:** run `simulazione.R` — it generates data, fits PCA/FA/AE, computes alignment-aware metrics, and saves results for each replication/setting.

---

## What the script does

- **Data generation:** draws latent variables $Z$ and produces observed data $X$ via a sparse loading matrix.  
  - *Linear:* $X = ZW$  
  - *Non-linear:* blocks based on $Z^3$ and $\exp(0.5Z)$, standardized before mixing.

- **Models:**  
  - PCA (top-$d$ components)  
  - FA (ML, varimax; scores by regression)  
  - AE (symmetric MLP with `tanh`, linear code layer, early stopping & ReduceLROnPlateau)

- **Metrics (alignment-aware):**  
  - Average **squared factor correlation** after best permutation/sign  
  - **MSE** after best permutation/sign  
  - **MSE** after **orthogonal Procrustes** rotation  

- **Grid & replications:** loops over $n \in \{1000, 10000\}$, $d \in \{2,5\}$, $p \in \{10,100,1000\}$, noise $\in$ {low, high}, transformation $\in$ {linear, nonlinear}, with `n_replications = 50`.

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
  ```

- For the autoencoder, you'll also need to install TensorFlow:
  ```r
  keras::install_keras()
  ```

---

## Mathematical Details

### Data Generation Process

The synthetic data follows this general structure:
- Latent factors: $Z \sim \mathcal{N}(0, I_d)$ where $d$ is the number of true factors
- Loading matrix: $W \in \mathbb{R}^{d \times p}$ (sparse, with blocks of non-zero loadings)

**Linear case:** 
$$X = ZW + \epsilon$$

**Non-linear case:**
$$X = f(Z)W + \epsilon$$
where $f(Z)$ applies element-wise transformations like $z^3$ and $\exp(0.5z)$ to different blocks of $Z$.

### Evaluation Metrics

1. **Factor Correlation:** $\text{corr}(\hat{Z}, Z)^2$ averaged after optimal permutation and sign flips
2. **Procrustes MSE:** $\|\hat{Z} - ZQ\|_F^2$ where $Q$ is the optimal orthogonal transformation

All alignment-aware metrics use the Hungarian algorithm to find optimal factor correspondences.

---

## Usage

Simply run the main script:
```r
source("simulazione.R")
```

The script will:
1. Generate synthetic datasets for each parameter combination
2. Fit PCA, FA, and AE models to each dataset
3. Compute all evaluation metrics
4. Save results to `risultati_simulazione/`
5. Print progress information during execution

Results can be loaded and analyzed using standard R data manipulation tools.
