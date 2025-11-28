# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def get_changepoint_matrix(t, t_change):
    """Assumes t and t_change are sorted."""
    T = len(t)
    S = len(t_change)
    A = np.zeros((T, S))
    for j in range(S):
        A[:, j] = (t >= t_change[j]).astype(float)
    return A


def compute_logistic_gamma(k, m, delta, t_change):
    """Adjusted offsets for piecewise continuity."""
    S = len(t_change)
    t_change_tensor = pt.as_tensor_variable(t_change)
    k_cumsum = pt.cumsum(delta)
    k_s = pt.concatenate([[k], k + k_cumsum])

    def gamma_step(i, gamma_cumsum, k_s, t_change_tensor, m):
        gamma_i = (t_change_tensor[i] - m - gamma_cumsum) * (1 - k_s[i] / k_s[i + 1])
        return gamma_cumsum + gamma_i, gamma_i

    (_, gammas), _ = pt.scan(
        fn=gamma_step,
        sequences=[pt.arange(S)],
        outputs_info=[pt.zeros(()), None],
        non_sequences=[k_s, t_change_tensor, m],
    )

    return gammas


def model(data):
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

    A = get_changepoint_matrix(t, t_change)

    with pm.Model() as pymc_model:
        # Priors
        k = pm.Normal("k", mu=0, sigma=5)
        m = pm.Normal("m", mu=0, sigma=5)
        delta = pm.Laplace("delta", mu=0, b=tau, shape=S)
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.5)
        beta = pm.Normal("beta", mu=0, sigma=sigmas, shape=K)

        # Trend
        if trend_indicator == 0:
            # Linear trend
            rate = k + pt.dot(A, delta)
            offset = m + pt.dot(A, -t_change * delta)
            trend = rate * t + offset
        else:
            # Logistic trend
            gamma = compute_logistic_gamma(k, m, delta, t_change)
            trend = cap * pm.math.sigmoid(
                (k + pt.dot(A, delta)) * (t - (m + pt.dot(A, gamma)))
            )

        # Likelihood
        mu = trend * (1 + pt.dot(X, beta * s_m)) + pt.dot(X, beta * s_a)
        _ = pm.Normal("y", mu=mu, sigma=sigma_obs, observed=y)

    return pymc_model
