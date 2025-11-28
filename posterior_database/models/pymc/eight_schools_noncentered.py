import numpy as np
import pymc as pm


def model(data):
    J = data["J"]  # number of schools
    y_obs = np.array(data["y"])  # estimated treatment
    sigma = np.array(data["sigma"])  # std of estimated effect
    with pm.Model() as pymc_model:
        mu = pm.Normal(
            "mu", mu=0, sigma=5
        )  # hyper-parameter of mean, non-informative prior
        tau = pm.HalfCauchy("tau", beta=5)  # hyper-parameter of sd
        theta_trans = pm.Normal("theta_trans", mu=0, sigma=1, shape=J)
        theta = mu + tau * theta_trans
        _ = pm.Normal("y", mu=theta, sigma=sigma, observed=y_obs)
    return pymc_model
