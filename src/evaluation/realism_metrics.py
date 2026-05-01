import numpy as np


def average_cross_asset_correlation_error(real_windows: np.ndarray, generated_windows: np.ndarray) -> float:
    real_flat = real_windows.reshape(-1, real_windows.shape[-1])
    generated_flat = generated_windows.reshape(-1, generated_windows.shape[-1])

    real_corr = np.corrcoef(real_flat, rowvar=False)
    generated_corr = np.corrcoef(generated_flat, rowvar=False)

    return float(np.mean(np.abs(real_corr - generated_corr)))


def average_volatility_error(real_windows: np.ndarray, generated_windows: np.ndarray) -> float:
    real_volatility = real_windows.reshape(-1, real_windows.shape[-1]).std(axis=0)
    generated_volatility = generated_windows.reshape(-1, generated_windows.shape[-1]).std(axis=0)

    return float(np.mean(np.abs(real_volatility - generated_volatility)))


def left_tail_quantile_error(
    real_windows: np.ndarray,
    generated_windows: np.ndarray,
    quantile: float = 0.01,
) -> float:
    real_tail = np.quantile(real_windows.reshape(-1, real_windows.shape[-1]), quantile, axis=0)
    generated_tail = np.quantile(generated_windows.reshape(-1, generated_windows.shape[-1]), quantile, axis=0)

    return float(np.mean(np.abs(real_tail - generated_tail)))


def kurtosis_error(real_windows: np.ndarray, generated_windows: np.ndarray) -> float:
    def kurtosis(x: np.ndarray) -> np.ndarray:
        centered = x - x.mean(axis=0)
        variance = np.mean(centered ** 2, axis=0)
        fourth_moment = np.mean(centered ** 4, axis=0)
        return fourth_moment / (variance ** 2 + 1e-12)

    real_flat = real_windows.reshape(-1, real_windows.shape[-1])
    generated_flat = generated_windows.reshape(-1, generated_windows.shape[-1])

    return float(np.mean(np.abs(kurtosis(real_flat) - kurtosis(generated_flat))))
