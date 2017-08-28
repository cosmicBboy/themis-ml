# Themis ML

[![Build Status](https://travis-ci.org/cosmicBboy/themis-ml.svg?branch=master)](https://travis-ci.org/cosmicBboy/themis-ml)

`themis-ml` is a Python library built on top of `pandas` and `sklearn`that
implements fairness-aware machine learning algorithms.

# Fairness-aware Techniques

`themis-ml` defines discrimination as the preference (bias) for or against a
set of social groups that result in the unfair treatment of its members with
respect to some opportunity or outcome.

`themis-ml` defines fairness as the inverse of discrimination, and in the
context of a machine learning algorithm, it is measured by the degree to which
the algorithm's predictions favor one social group over another in relation to
an outcome that holds socioeconomic, political, or legal importance, e.g.
the denial/approval of a loan application.

According to these definitions, a perfectly fair algorithm is one in which
the proportion of approved loans among minorities is equal to the proportion
of approved loans among white people.

## Measuring Discrimination

- [ ] Mean difference
- [ ] Normalized mean difference
- [ ] Consistency
- [ ] Situation Test Score

## Mitigating Discrimination

### Preprocessing

- [ ] Massaging (Relabelling)
- [ ] Reweighting
- [ ] Sampling

### Model Estimation

- [ ] Additive Counterfactually Fair Estimator
- [ ] Prejudice Remover Regularized Estimator

### Postprocessing

- [ ] Reject Option Classification
- [ ] Discrimination-aware Ensemble Classification


## Datasets

- [ ] German Credit Dataset
