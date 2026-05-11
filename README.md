# Probabilistic Deep Learning for Oceanographic Interval Estimation: A Comparative Study of Tube Loss

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch/TensorFlow](https://img.shields.io/badge/Framework-PyTorch_|_TensorFlow-orange.svg)]()

## 📌 Project Overview
This repository contains the code, models, and analysis for predicting **Significant Wave Height (SWH)** and **Sea Level Anomaly (SLA)** across the Indian Ocean. The core focus of this research is comparing the efficacy of standard probabilistic forecasting methods (like Quantile/Pinball Loss, DeepAR, and Mixture Density Networks) against a novel, distribution-free interval estimation approach known as **Tube Loss**.

Traditional parametric models often fail on highly skewed, volatile oceanographic data (e.g., tropical cyclones). This project demonstrates that applying the Tube Loss objective function to efficient sequence models produces sharper, more stable prediction intervals while reducing computational overhead.

## 🌊 Dataset & Physical Forcings
The study utilizes 1096 days of daily CMEMS data across five distinct geographical coordinates in the Indian Ocean, each characterized by unique physical forcings:
* **Arabian Sea:** Dominated by the highly seasonal Southwest Monsoon (Findlater Jet).
* **Bay of Bengal:** Highly volatile, driven by freshwater influx and severe tropical cyclones.
* **Andaman Sea:** Semi-enclosed basin with localized tidal forcing and Irrawaddy eddies.
* **Lakshadweep Sea:** Influenced by planetary Rossby waves and the mini warm pool.
* **Southern Indian Ocean:** Governed by continuous Southeast Trade Winds.

> **Note on Pre-processing:** Due to the extreme outliers present in storm data, `RobustScaler` (Median + IQR) is used instead of standard mean-variance scaling. This prevents massive cyclone spikes from washing out the baseline data distribution.

## 🧠 Architectures Evaluated
Nine distinct baseline and advanced architectures were evaluated using a 30-day sequence length (`SEQ_LEN=30`).

**Deterministic / Sequence Backbones:**
1. **SimpleRNN:** Baseline for temporal memory.
2. **LSTM:** Utilizes a Cell State to maintain long-term physical trends (highly effective for SLA).
3. **GRU:** Efficient gated mechanism forcing a strict update trade-off (highly effective for SWH).
4. **Transformer:** Utilizes Self-Attention to dynamically route distant storm features.
5. **Mamba (SSM):** Features an input-dependent selective state space, acting as an adaptive physical noise filter.

**Probabilistic Baselines:**
6. **DeepAR:** Parametric baseline predicting a symmetrical Gaussian distribution (struggles with skewed storm data).
7. **MDN-GRU:** Predicts a mixture of Gaussians to handle multi-modal distributions (prone to mode collapse).
8. **LSTM-KDE:** Non-parametric baseline using Kernel Density Estimation on residuals.
9. **ARIMA:** Classical statistical time-series baseline.

## 📉 The Tube Loss Function
Instead of relying on two separate models for upper and lower quantiles, or forcing a shape assumption like DeepAR, this repository implements **Tube Loss**, defined as:

$$
\min_{(\mu_{1},\mu_{2})} \left[ \sum_{i=1}^{m}\rho_{t}^{r}(y_{i},\mu_{1}(x_{i}),\mu_{2}(x_{i})) + \delta\sum_{i=1}^{m}|\mu_{2}(x_{i})-\mu_{1}(x_{i})| \right]
$$

This function natively penalizes wide intervals via the $\delta$ parameter and allows the interval to shift vertically to capture skewed data clusters via the $r$ asymmetry parameter.

## 📊 Key Findings
* **SWH Winner (GRU/Mamba + Tube Loss):** For highly volatile, wind-driven wave heights, computationally lighter models like GRU or noise-filtering models like Mamba paired with the distribution-free Tube Loss captured sharp storm spikes without blowing up the Mean Prediction Interval Width (MPIW).
* **SLA Winner (LSTM + Tube Loss):** For slow-moving, accumulated physical processes like Sea Level Anomaly, the LSTM's robust Cell State combined with Tube Loss provided the most stable interval estimations.
* **Methodological Robustness:** All experiments were run across **5 distinct random seeds** to ensure mathematical stability and rule out stochastic variance. Tube Loss consistently demonstrated lower standard deviation across runs compared to unstable Negative Log-Likelihood (NLL) optimizations like MDNs.

## 📂 Repository Structure
```text
├── data/                   # Raw and processed CMEMS data files (.xlsx, .csv)
├── notebooks/              # Jupyter notebooks containing EDA and evaluation heatmaps
├── src/
│   ├── data_loader.py      # Pre-processing, gap interpolation, and RobustScaler logic
│   ├── models/             # PyTorch/TensorFlow implementations of all 9 architectures
│   ├── losses.py           # Custom implementations of Pinball Loss and Tube Loss
│   └── train.py            # Training loop with multi-seed execution
├── analysis_plots/         # Output directory for reliability diagrams and heatmaps
├── requirements.txt        # Python dependencies
└── README.md
