"""
Prophet: Structural Time Series Model

A PyMC implementation of Facebook's Prophet model for time series forecasting.
Supports both linear and logistic (saturating) growth trends with changepoints,
plus additive and multiplicative seasonality via Fourier features.

Reference: Taylor & Letham (2018). Forecasting at Scale. The American Statistician.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def get_changepoint_matrix(t, t_change):
    """
    Create changepoint indicator matrix A.

    Parameters
    ----------
    t : array of shape (T,)
        Time points (normalized)
    t_change : array of shape (S,)
        Times of trend changepoints

    Returns
    -------
    A : array of shape (T, S)
        A[i,j] = 1 if t[i] >= t_change[j], else 0
    """
    T = len(t)
    S = len(t_change)
    A = np.zeros((T, S))
    for j in range(S):
        A[:, j] = (t >= t_change[j]).astype(float)
    return A


def compute_logistic_gamma(k, m, delta, t_change):
    """
    Compute gamma offsets for piecewise logistic trend continuity.

    Uses pt.scan to vectorize the sequential computation where each
    gamma[i] depends on the cumulative sum of gamma[0:i].

    Parameters
    ----------
    k : scalar tensor
        Base growth rate
    m : scalar tensor
        Trend offset
    delta : tensor of shape (S,)
        Rate adjustments at changepoints
    t_change : array of shape (S,)
        Times of changepoints

    Returns
    -------
    gamma : tensor of shape (S,)
        Offset adjustments for continuity at each changepoint
    """
    S = len(t_change)
    t_change_tensor = pt.as_tensor_variable(t_change)

    # Cumulative rate: k_s[i] is the rate in segment i
    # k_s = [k, k + delta[0], k + delta[0] + delta[1], ...]
    k_cumsum = pt.cumsum(delta)
    k_s = pt.concatenate([[k], k + k_cumsum])

    def gamma_step(i, gamma_cumsum, k_s, t_change_tensor, m):
        """Compute gamma[i] given cumulative sum of previous gammas."""
        gamma_i = (t_change_tensor[i] - m - gamma_cumsum) * (1 - k_s[i] / k_s[i + 1])
        return gamma_cumsum + gamma_i, gamma_i

    # Use scan to compute gammas sequentially
    (_, gammas), _ = pt.scan(
        fn=gamma_step,
        sequences=[pt.arange(S)],
        outputs_info=[pt.zeros(()), None],
        non_sequences=[k_s, t_change_tensor, m],
    )

    return gammas


def model(data):
    """
    Create a PyMC model for Prophet-style time series forecasting.

    Parameters
    ----------
    data : dict
        Dictionary containing:
        - T: int, number of time periods
        - K: int, number of regressors (Fourier features)
        - t: array (T,), normalized time values
        - cap: array (T,), carrying capacity for logistic trend
        - y: array (T,), observed time series values
        - S: int, number of changepoints
        - t_change: array (S,), times of changepoints
        - X: array (T, K), regressor matrix (Fourier features)
        - sigmas: array (K,), prior scales for regressor coefficients
        - tau: float, prior scale for changepoint rate adjustments
        - trend_indicator: int, 0 for linear trend, 1 for logistic
        - s_a: array (K,), indicator of additive features
        - s_m: array (K,), indicator of multiplicative features

    Returns
    -------
    pymc_model : pm.Model
        PyMC model ready for sampling
    """
    # Extract data
    T = data["T"]
    K = data["K"]
    t = np.array(data["t"])
    cap = np.array(data["cap"])
    y = np.array(data["y"])
    S = data["S"]
    t_change = np.array(data["t_change"])
    X = np.array(data["X"])
    sigmas = np.array(data["sigmas"])
    tau = data["tau"]
    trend_indicator = data["trend_indicator"]
    s_a = np.array(data["s_a"])
    s_m = np.array(data["s_m"])

    # Compute changepoint matrix (transformed data in Stan)
    A = get_changepoint_matrix(t, t_change)

    with pm.Model() as pymc_model:
        # Priors (matching Stan model)
        k = pm.Normal("k", mu=0, sigma=5)  # Base trend growth rate
        m = pm.Normal("m", mu=0, sigma=5)  # Trend offset
        delta = pm.Laplace("delta", mu=0, b=tau, shape=S)  # Rate adjustments
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.5)  # Observation noise
        beta = pm.Normal("beta", mu=0, sigma=sigmas, shape=K)  # Regressor coefficients

        # Compute trend based on indicator
        if trend_indicator == 0:
            # Linear trend: (k + A @ delta) * t + (m + A @ (-t_change * delta))
            rate = k + pt.dot(A, delta)
            offset = m + pt.dot(A, -t_change * delta)
            trend = rate * t + offset
        else:
            # Logistic trend with carrying capacity
            gamma = compute_logistic_gamma(k, m, delta, t_change)
            trend = cap * pm.math.sigmoid(
                (k + pt.dot(A, delta)) * (t - (m + pt.dot(A, gamma)))
            )

        # Seasonality components
        seasonality_multiplicative = pt.dot(X, beta * s_m)
        seasonality_additive = pt.dot(X, beta * s_a)

        # Combined mean: trend * (1 + multiplicative) + additive
        mu = trend * (1 + seasonality_multiplicative) + seasonality_additive

        # Likelihood
        _ = pm.Normal("y", mu=mu, sigma=sigma_obs, observed=y)

    return pymc_model
