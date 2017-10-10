# Themis ML

[![Build Status](https://travis-ci.org/cosmicBboy/themis-ml.svg?branch=master)](https://travis-ci.org/cosmicBboy/themis-ml)
[![Documentation Status](https://readthedocs.org/projects/themis-ml/badge/?version=latest)](http://themis-ml.readthedocs.io/en/latest/?badge=latest)

<img src="static/themis-ml-logo.svg" width="100" height="100">

`themis-ml` is a Python library built on top of `pandas` and `sklearn`that
implements fairness-aware machine learning algorithms.

# Fairness-aware Machine Learning

`themis-ml` defines discrimination as the preference (bias) for or against a
set of social groups that result in the unfair treatment of its members with
respect to some outcome.

It defines fairness as the inverse of discrimination, and in the context of a
machine learning algorithm, this is measured by the degree to which the
algorithm's predictions favor one social group over another in relation to an
outcome that holds socioeconomic, political, or legal importance, e.g. the
denial/approval of a loan application.

A "fair" algorithm depends on how we define fairness. For example if we define
fairness as statistical parity, a fair algorithm is one in which the proportion
of approved loans among minorities is equal to the proportion of approved loans
among white people.

# Features

Here are a few of the discrimination discovery and fairness-aware techniques
that this library implements.

### Measuring Discrimination

- [X] Mean difference
- [X] Normalized mean difference
- [ ] Consistency
- [ ] Situation Test Score

### Mitigating Discrimination

#### Preprocessing

- [X] Relabelling (Massaging)
- [ ] Reweighting
- [ ] Sampling

#### Model Estimation

- [X] Additive Counterfactually Fair Estimator
- [ ] Prejudice Remover Regularized Estimator

#### Postprocessing

- [X] Reject Option Classification
- [ ] Discrimination-aware Ensemble Classification

### Datasets

`themis-ml` also provides utility functions for loading freely available
datasets from a variety of sources.

- [X] German Credit [(source)][german-credit]
- [X] Census Income [(source)][census-income]
- [ ] Taiwan Credit Default [(source)][taiwan-credit]
- [ ] Australian Credit Approval [(source)][australian-credit]
- [ ] Adult Census [(source)][adult-census]
- [ ] Communities and Crime [(source)][communities-crime]
- [ ] Disabled Residents Expenditure [(source)][disabled-expenditure]

# Installation

The source code is currently hosted on GitHub: https://github.com/cosmicBboy/themis-ml.
You can install the latest released version with `conda` or `pip`.

```
# conda
conda install themis-ml
```

```
#pip
pip install themis-ml
```

# Documentation

Official documentation for this package can be found [here][docs]

# References

You can find a complete set of references for the discrimination discovery and
fairness-aware methods implemented in `themis-ml` in this [paper](paper/main.pdf).

[german-credit]: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
[taiwan-credit]: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
[australian-credit]: https://archive.ics.uci.edu/ml/datasets/Statlog+%28Australian+Credit+Approval%29
[adult-census]: https://archive.ics.uci.edu/ml/datasets/Adult
[census-income]: https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29
[communities-crime]: https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime
[disabled-expenditure]: http://ww2.amstat.org/publications/jse/v22n1/mickel.pdf
[docs]: http://themis-ml.readthedocs.io/en/latest/
