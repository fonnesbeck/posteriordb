# `posteriordb`: a database of Bayesian posterior inference

> This repository is a fork of [stan-dev/posteriordb](https://github.com/stan-dev/posteriordb) that integrates the Python API from [stan-dev/posteriordb-python](https://github.com/stan-dev/posteriordb-python) directly into the main repository. This provides a unified package containing both the database content and Python access tools.

## What is `posteriordb`?

`posteriordb` is a set of posteriors, i.e. Bayesian statistical models and data sets, reference implementations in probabilistic programming languages, and reference posterior inferences in the form of posterior samples.

## Why use `posteriordb`?

`posteriordb` is designed to test inference algorithms across a wide range of models and data sets. Applications include testing for accuracy, speed, and scalability. `posteriordb` can be used to test new algorithms being developed or deployed as part of continuous integration for ongoing regression testing algorithms in probabilistic programming frameworks.

`posteriordb` also makes it easy for students and instructors to access various pedagogical and real-world examples with precise model definitions, well-curated data sets, and reference posteriors.

For more details regarding the use cases of `posteriordb`, see [doc/use_cases.md](https://github.com/fonnesbeck/posteriordb/blob/master/doc/use_cases.md).

## Installation

Install using pip:

```bash
pip install posteriordb
```

Or install from source:

```bash
git clone https://github.com/fonnesbeck/posteriordb
cd posteriordb
pip install .
```

For development, this project uses [Pixi](https://pixi.sh/):

```bash
pixi install
pixi run test
```

## Using the posterior database

The database provides convenience functions to access data, model code, and information for individual posteriors.

### Loading the database

For local access with a cloned repository:

```python
from posteriordb import PosteriorDatabase

pdb = PosteriorDatabase("posterior_database")
```

### Listing available content

```python
# List all posteriors
>>> pdb.posterior_names()[:5]
['GLMM_Poisson_data-GLMM_Poisson_model',
 'GLMM_data-GLMM1_model',
 'GLM_Binomial_data-GLM_Binomial_model',
 'GLM_Poisson_Data-GLM_Poisson_model',
 'M0_data-M0_model']

# List all models
>>> pdb.model_names()[:5]
['2pl_latent_reg_irt',
 'GLMM1_model',
 'GLMM_Poisson_model',
 'GLM_Binomial_model',
 'GLM_Poisson_model']

# List all datasets
>>> pdb.data_names()[:5]
['GLMM_Poisson_data',
 'GLMM_data',
 'GLM_Binomial_data',
 'GLM_Poisson_Data',
 'M0_data']
```

### Accessing posteriors

The posterior's name is made up of the data and model fitted to the data. Together, these two uniquely define a posterior distribution.

```python
>>> posterior = pdb.posterior("eight_schools-eight_schools_noncentered")
>>> posterior.name
'eight_schools-eight_schools_noncentered'

# Access the model and data
>>> model = posterior.model
>>> data = posterior.data

>>> model.name
'eight_schools_noncentered'

>>> data.name
'eight_schools'
```

You can also access models and datasets directly:

```python
>>> model = pdb.model("eight_schools_noncentered")
>>> data = pdb.data("eight_schools")
```

### Working with models

Access model code for different probabilistic programming languages:

```python
>>> print(model.code("stan"))
data {
  int<lower=0> J; // number of schools
  array[J] real y; // estimated treatment
  array[J] real<lower=0> sigma; // std of estimated effect
}
parameters {
  vector[J] theta_trans; // transformation of theta
  real mu; // hyper-parameter of mean
  real<lower=0> tau; // hyper-parameter of sd
}
transformed parameters {
  vector[J] theta;
  // original theta
  theta = theta_trans * tau + mu;
}
model {
  theta_trans ~ normal(0, 1);
  y ~ normal(theta, sigma);
  mu ~ normal(0, 5); // a non-informative prior
  tau ~ cauchy(0, 5);
}

# Get the path to the model file
>>> model.code_path("stan")
'posterior_database/models/stan/eight_schools_noncentered.stan'

# List available frameworks for this model
>>> model.frameworks
['stan', 'pymc']

# Get model information
>>> model.information
{'name': 'eight_schools_noncentered',
 'title': 'A non-centered hiearchical model for 8 schools',
 'description': 'A non-centered hiearchical model for the 8 schools example of Rubin (1981)',
 'keywords': ['bda3_example', 'hiearchical'],
 'references': ['rubin1981estimation', 'gelman2013bayesian'],
 ...}
```

Note that the references are BibTeX keys that can be found in `posterior_database/bibliography/references.bib`.

### Working with data

```python
>>> data.values()
{'J': 8,
 'y': [28, 8, -3, 7, -1, 1, 18, 12],
 'sigma': [15, 10, 16, 11, 9, 11, 10, 18]}

>>> data.information
{'name': 'eight_schools',
 'keywords': ['bda3_example'],
 'title': 'The 8 schools dataset of Rubin (1981)',
 'description': 'A study for the Educational Testing Service to analyze the effects of special coaching programs on test scores.',
 'references': ['rubin1981estimation', 'gelman2013bayesian'],
 ...}
```

### Reference posterior draws

Access gold standard posterior draws for algorithm validation:

```python
>>> posterior.reference_draws_info()
{'name': 'eight_schools-eight_schools_noncentered',
 'inference': {'method': 'stan_sampling',
  'method_arguments': {'chains': 10,
   'iter': 20000,
   'warmup': 10000,
   'thin': 10,
   'seed': 4711,
   'control': {'adapt_delta': 0.95}}},
 'diagnostics': {...}}

# Get the draws (list of dicts, one per chain)
>>> draws = posterior.reference_draws()
>>> len(draws)
10

# Convert to DataFrame
>>> import pandas as pd
>>> pd.DataFrame(draws[0])
```

### Posterior aliases

Common posteriors have short aliases:

```python
# "eight_schools" resolves to "eight_schools-eight_schools_noncentered"
>>> posterior = pdb.posterior("eight_schools")
>>> posterior.name
'eight_schools-eight_schools_noncentered'
```

## Content

See [DATABASE_CONTENT.md](https://github.com/fonnesbeck/posteriordb/blob/master/doc/DATABASE_CONTENT.md) for the details content of the posterior database.

## Contributing

We are happy with any help in adding posteriors, data, and models to the database! See [CONTRIBUTING.md](https://github.com/fonnesbeck/posteriordb/blob/master/doc/CONTRIBUTING.md) for details on how to contribute.

## Design choices

The main focus of the database is simplicity, both in understanding and in use.

1. **Priors are hardcoded in model files** - Changing the prior changes the posterior. Create a new model to test different priors.
2. **Data transformations are stored as different datasets** - Create new data to test different data transformations, subsets, and variable settings.
3. **Models and data have info.json files** - Model and data specific information is stored in `(model/data).info.json` files.
4. **Prefix `syn_` indicates synthetic data** - The generative process is known and found in `data-raw`.
5. **All data preprocessing is included** - Found in `posterior_database/data/data-raw`.
6. **PPL-specific information goes in code comments** - Not in `model.info.json` files.

## Versioning of models

Models may be updated over time. However, models will only have the same name in posteriordb if the log density is the same (up to a normalizing constant). Otherwise, a new model will be added to the database.
