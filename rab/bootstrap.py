import numpy as np
from typing import Optional
from scipy.optimize import curve_fit

EXP_FUNC = lambda x, A, p: A * p ** np.array(x)


def fit_exp(x, data, *args, **kwargs):
    return curve_fit(EXP_FUNC, x, data, **kwargs)


def data_uncertainties(data, confidence: int, data_median: Optional[np.ndarray] = None):
    """Compute the uncertainties of the median (or specified) values.

    Args:
        data (list or np.ndarray): 2d array with rows containing data points
            from which the median value is extracted.
        confidence (int): between 0 and 100, the quantile confidence interval

    Returns:
        np.ndarray: uncertainties of the data.
    """

    percentiles = [
        (100 - confidence) / 2,
        (100 + confidence) / 2,
    ]
    if data_median is None:
        data_median = np.median(data, axis=1)
    percentile_inteval = np.percentile(data, percentiles, axis=1)
    uncertainties = np.abs(np.vstack([data_median, data_median]) - percentile_inteval)
    return uncertainties


def bootstrap(data: np.ndarray, nbootstraps: int, seed: int = None) -> np.ndarray:
    """Non-parametric bootstrap resampling.

    Args:
        data (np.ndarray): 2d array with ROWS containing samples. For m datapoints with niter iterations shape: (m, niter).

    Returns:
        np.ndarray: resampled data of shape (*data.shape, nbootstraps)
    """

    random_generator = np.random.default_rng(seed)
    sample_size = len(data[0])
    random_inds = random_generator.integers(
        0, sample_size, size=(sample_size, nbootstraps)
    )
    return np.array(data)[:, random_inds]


def fit_with_bootstrap(
    x: np.ndarray, y_scatter: np.ndarray, nbootstraps: int, confidence: int
):
    # Non-parametric bootstrap resampling
    bootstrap_y = bootstrap(y_scatter, nbootstraps)
    # Compute y and popt estimates for each bootstrap iteration
    y_estimates = np.mean(bootstrap_y, axis=1)
    popt_estimates = np.apply_along_axis(
        lambda y_iter: fit_exp(x, y_iter, bounds=[-1, 1])[0],
        axis=0,
        arr=np.array(y_estimates),
    )
    # Fit the initial data and compute error bars
    y_averages = [np.mean(y_row) for y_row in y_scatter]
    error_bars = data_uncertainties(y_estimates, confidence)
    # sigma = np.max(error_bars, axis=0) + 0.1
    popt, pcov1 = fit_exp(x, y_averages)  # , sigma=sigma, bounds=[0, 1])
    pcov = data_uncertainties(popt_estimates, confidence, data_median=popt)
    return x, y_averages, error_bars, popt, pcov.T
