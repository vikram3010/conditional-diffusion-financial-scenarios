# Project Summary

## Title

Counterfactual Financial Scenario Generation Using Conditional Diffusion Models

## Objective

The goal of this project is to generate realistic multivariate financial return scenarios using a conditional denoising diffusion probabilistic model. The model is conditioned on macroeconomic variables so that generated scenarios can reflect different market regimes and economic contexts.

## Problem

Traditional financial simulation methods such as Geometric Brownian Motion often struggle to capture complex real-world properties such as cross-asset dependence, volatility behavior, tail risk, and regime sensitivity. This project explores a deep generative modeling approach for producing more realistic synthetic return windows.

## Data

The project uses daily financial return data for:

- SPY
- QQQ
- GLD

The return data is aligned with macroeconomic conditioning variables including:

- Effective Federal Funds Rate
- 10-Year Treasury Yield
- CPI
- Unemployment Rate
- VIX

The aligned modeling period covers 2010–2023.

## Methodology

The pipeline includes:

- Market and macroeconomic data collection
- Strict date alignment to avoid leakage
- Rolling-window sequence generation
- Time-based train, validation, and test splitting
- Conditional DDPM model design in PyTorch
- Temporal convolutional residual architecture
- Timestep embeddings and macroeconomic conditioning
- Baseline comparison against Geometric Brownian Motion
- Evaluation using distributional, volatility, correlation, and tail-risk metrics

## Model

The conditional DDPM predicts noise added to financial return windows at different diffusion timesteps. The model receives:

- A noisy return window
- A macroeconomic conditioning window
- A diffusion timestep

It outputs the predicted noise used for denoising and generation.

## Evaluation Metrics

The project evaluates generated scenarios using:

- Noise-prediction MSE
- Cross-asset correlation error
- Volatility error
- Left-tail quantile error
- Kurtosis error
- Autocorrelation behavior
- Distributional realism checks

## Key Skills Demonstrated

- Deep learning
- Time-series modeling
- Financial data analysis
- PyTorch model development
- Data preprocessing
- Leakage-aware train/test splitting
- Statistical evaluation
- Generative AI
