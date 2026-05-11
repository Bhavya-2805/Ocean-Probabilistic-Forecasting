# Probabilistic Deep Learning for Oceanographic Interval Estimation: Tube Loss vs. Quantile Regression

## 📌 Project Overview
This research investigates advanced probabilistic deep learning architectures for forecasting complex oceanographic variables across the Indian Ocean: **Significant Wave Height (SWH)** and **Sea Level Anomaly (SLA)**. 

The core contribution of this repository is a rigorous, multi-architecture comparison of traditional uncertainty quantification methods against **Tube Loss**, a novel, distribution-free objective function for direct Prediction Interval (PI) estimation. By mapping mathematical model behaviors directly to physical ocean forcings, this project identifies the optimal pairings of sequence backbones and loss functions for highly volatile vs. slow-moving environments.

## 🌊 Dataset & Physical Forcings
The study utilizes 1096 days of daily CMEMS (Copernicus Marine Environment Monitoring Service) data across five distinct geographical coordinates, each presenting unique forecasting challenges:
* **Arabian Sea (15.0°N, 65.0°E):** Dominated by the highly seasonal Southwest Monsoon (Findlater Jet).
* **Bay of Bengal (15.0°N, 90.0°E):** Highly volatile basin driven by massive freshwater influx and severe, sudden tropical cyclones.
* **Andaman Sea (10.0°N, 95.0°E):** Semi-enclosed basin influenced by localized tidal forcing and Irrawaddy eddies.
* **Lakshadweep Sea (10.0°N, 72.5°E):** Governed by slow-moving planetary Rossby waves radiating from the Indian coast.
* **Southern Indian Ocean (-10.0°N, 75.0°E):** Constantly noisy but lower-variance region driven by Southeast Trade Winds.

> **Pre-processing Strategy:** To prevent data leakage while preserving the physical reality of extreme weather events, the pipeline utilizes a `RobustScaler` (Median + Interquartile Range) fitted strictly on the training set, preventing massive cyclone spikes from artificially washing out baseline data.

## 🧠 Sequence Architectures Evaluated
Nine models were evaluated using a strict 30-day lookback window (`SEQ_LEN = 30`).

**Deterministic / Sequence Backbones:**
1. **SimpleRNN:** The baseline for temporal memory.
2. **LSTM:** Utilizes a "Cell State" highway to maintain long-term physical trends without fading.
3. **GRU:** Merges gates for computational efficiency, forcing a strict update/forget trade-off.
4. **Transformer:** Replaces sequential reading with Self-Attention, allowing direct routing of distant features.
5. **Mamba (SSM):** A Structured State Space Model featuring an input-dependent selective scan mechanism.

**Statistical & Probabilistic Baselines:**
6. **ARIMA:** Classical statistical time-series baseline.
7. **DeepAR:** Parametric autoregressive model predicting the mean and variance of a forced Gaussian distribution.
8. **MDN-GRU:** Predicts a mixture of $K$ Gaussians to handle multi-modal wave distributions.
9. **LSTM-KDE:** Non-parametric baseline using Kernel Density Estimation on model residuals.

## 📉 Loss Functions for Uncertainty Quantification

### 1. The Standard: Quantile (Pinball) Loss
The traditional approach to probabilistic forecasting is Quantile Regression, utilizing the asymmetric Pinball Loss function: $\rho_q(e) = \max(q \cdot e, (q-1) \cdot e)$.
While effective, generating a 95% prediction interval mathematically requires training two separate neural network bounds, doubling computational overhead.

### 2. The Novel Approach: Tube Loss
This repository implements **Tube Loss**, a direct interval estimation method defined by a piecewise function combined with a strict architectural penalty:

$$
\min_{(\mu_{1},\mu_{2})} \left[ \sum_{i=1}^{m}\rho_{t}^{r}(y_{i},\mu_{1}(x_{i}),\mu_{2}(x_{i})) + \delta\sum_{i=1}^{m}|\mu_{2}(x_{i})-\mu_{1}(x_{i})| \right]
$$

It computes the upper and lower bounds simultaneously in a single pass. The $\delta$ parameter explicitly forces the neural network to minimize the interval width, while the $r$ parameter allows the entire "tube" to shift vertically to capture highly skewed, asymmetrical storm data.

## 📊 Empirical Findings & Results

All models were evaluated across **5 distinct random seeds** to eliminate stochastic variance and prove mathematical stability. 

**1. Pareto-Optimality in Sea Level Anomaly (SLA)**
* **GRU-Quantile and SimpleRNN-Quantile** emerged as Pareto-optimal for SLA forecasting. They achieved a Prediction Interval Coverage Probability (PICP) $\ge$ 95% at 4–5 locations while maintaining a mean Prediction Interval Width (MPIW) < 0.012 m. 
* **SimpleRNN-Quantile** proved to be the most reproducible architecture overall, passing the 95% coverage target at all five locations with an incredibly stable standard deviation (`std(PICP) = 1.09%`) across all random seeds.

**2. The Impact of Physical Forcing on SWH Uniformity**
* Calibration for Significant Wave Height (SWH) was markedly more uniform across architectures than SLA. 
* **62% of model-location pairs** reached PICP $\ge$ 95% for SWH (versus only 51% for SLA), and **65% fell into the optimal 94–97% "sweet spot"** (versus only 22% for SLA). 
* This improved uniformity reflects the stronger, more regular monsoon forcing in SWH, which provides a dominant physical signal that all architectures can easily learn.

**3. The Over-Parameterization Paradox (Transformer & Mamba)**
* High-complexity models (**Transformer and Mamba**) catastrophically failed on SLA but successfully recovered on SWH. For example, Transformer-Tube achieved a dismal 46.6% PICP at the Southern IO for SLA, but a highly accurate 95.5% PICP at the Andaman Sea for SWH. 
* **The mechanism for this paradox is three-fold:**
  1. SWH has a 20× higher signal amplitude than SLA, giving complex models a clearer target.
  2. SWH possesses a dominant annual monsoon periodicity that is largely absent in SLA dynamics.
  3. The massive parameter-to-data ratio ($\approx 10^7$) places both Transformer and Mamba in a severely over-parameterized regime for the relatively small 846-sequence SLA training set, leading to severe overfitting on the slow-moving SLA trends.

**4. Computational Efficiency of Tube Loss**
* By estimating bounds simultaneously, **Tube Loss reduced computational training overhead by roughly 50%** compared to the dual-pass Quantile Regression method. 
* Multi-seed evaluation proved that Tube Loss provides a vastly smoother optimization landscape than Negative Log-Likelihood (NLL) methods like MDN-GRU, avoiding mode-collapse instability.

