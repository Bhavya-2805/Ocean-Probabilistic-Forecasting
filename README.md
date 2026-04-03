# 🌊 Ocean Probabilistic Forecasting

This repository contains the codebase for advanced deep learning models designed to generate probabilistic forecasts for oceanographic variables, specifically **Sea Level Anomaly (SLA)** and **Significant Wave Height (SWH)**. 

Unlike standard deterministic models that output a single point prediction, this project utilizes specialized loss functions to generate **Prediction Intervals (Tubes)**, capturing the inherent chaos and uncertainty of ocean dynamics across the Indian Ocean.

## 🎯 Project Objectives
* **Probabilistic Forecasting:** Predicting future sea level and wave height states with mathematically rigorous upper and lower confidence bounds.
* **Multi-Region Analysis:** Forecasting across highly volatile sub-regions of the Indian Ocean, including:
  * The Arabian Sea
  * The Bay of Bengal
  * The Andaman Sea
  * The Equatorial Indian Ocean
* **Algorithm Optimization:** Comparing the temporal feature extraction capabilities of Recurrent Neural Networks (LSTM, GRU, SimpleRNN) on daily satellite data.

## 🧮 Mathematical Architecture
Standard Mean Squared Error (MSE) is insufficient for capturing ocean volatility. This codebase implements custom gradients in TensorFlow/Keras to optimize for prediction intervals:
1. **Quantile (Pinball) Loss:** Asymmetric penalty functions to mathematically bind predictions to specific percentiles (e.g., 15th and 85th).
2. **Custom Tube Loss:** A dynamic loss function that applies a width penalty ($\delta$) to aggressively shrink the forecasting envelope while maintaining strict data coverage constraints.

## 📊 Evaluation Metrics
To measure the success of the probabilistic forecasts against benchmark models (like DeepAR), we evaluate using:
* **PICP (Prediction Interval Coverage Probability):** The percentage of actual satellite observations successfully captured inside our forecasted tube.
* **MPIW (Mean Prediction Interval Width):** The average thickness of the predicted tube (optimized to be as tight as possible).

## 📂 Data Sources
Ground-truth testing and model training utilize daily satellite altimetry data provided by the **Copernicus Marine Environment Monitoring Service (CMEMS)**.

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow 2.x, Keras
* **Geospatial Processing:** xarray, netCDF4
* **Data Manipulation:** NumPy, Pandas, scikit-learn
* **Visualization:** Matplotlib
