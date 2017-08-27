# Contributing to `themis-ml`

## Development Environment

Create a `themis-ml` development environment as follows:

- install [Anaconda](https://www.continuum.io/downloads) or
  [miniconda](https://conda.io/miniconda.html)
- clone the [repository](https://github.com/cosmicBboy/themis-ml)
- `cd` into the `themis-ml` source directory

Create a new virtual environment with conda
(currently development in Python2.7 is supported):

```
conda create -n themis_ml_dev python=2.7 --file environment_dev.yml

# to update your current environment
conda update -n themis_ml_dev --file environment_dev.yml
```

To work in this environment, Mac OSX Linux users should:

```
source activate themis_ml_dev

# to deactivate:
source deactivate
```

Optionally, you can use `direnv` to automatically load the `themis_ml_dev`
virtual environment whenever you `cd` into the source repository.

```
# on MacOSX
brew install direnv

# in themis-ml source directory
direnv allow .
```
