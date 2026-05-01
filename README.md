# Conditional Diffusion Model for Financial Time-Series Scenario Generation

This project implements a conditional denoising diffusion probabilistic model (DDPM) for generating multivariate financial return scenarios conditioned on macroeconomic variables. The model generates synthetic return windows for SPY, QQQ, and GLD using macroeconomic context such as interest rates, Treasury yields, CPI, unemployment, and VIX.

## Project Overview

Financial time-series modeling requires capturing volatility, cross-asset dependence, tail behavior, and regime-sensitive dynamics. This project explores whether a conditional diffusion model can generate more realistic financial scenarios than a traditional statistical baseline.

The pipeline includes:

- Market and macroeconomic data collection
- Strict time-based data alignment
- Rolling-window construction
- Train, validation, and test splitting without look-ahead leakage
- Conditional DDPM model training in PyTorch
- Baseline comparison against Geometric Brownian Motion
- Evaluation using distributional, correlation, volatility, and tail-risk metrics

## Data

Target assets:

- SPY
- QQQ
- GLD

Macroeconomic conditioning variables:

- Effective Federal Funds Rate
- 10-Year Treasury Yield
- CPI
- Unemployment Rate
- VIX

The final aligned dataset covers daily observations from 2010 to 2023 and uses rolling windows for sequence modeling.

## Model

The main model is a conditional DDPM implemented in PyTorch. The architecture uses:

- Sinusoidal timestep embeddings
- Temporal convolutional layers
- Residual temporal blocks
- Macro-conditioning embeddings
- Noise-prediction objective
- Conditional generation over financial return windows

## Evaluation

The model is evaluated using metrics such as:

- Test noise-prediction MSE
- Cross-asset correlation error
- Volatility error
- Left-tail quantile error
- Kurtosis error
- Autocorrelation behavior
- Distributional realism checks

## Tech Stack

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- SciPy
- Time-Series Modeling
- Deep Learning
- Financial Data Analysis

## Repository Structure

```text
conditional-diffusion-financial-scenarios/
├── README.md
├── requirements.txt
├── notebooks/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── evaluation/
├── outputs/
│   ├── figures/
│   └── tables/
└── reports/
