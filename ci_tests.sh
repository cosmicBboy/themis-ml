#!/bin/bash
set -ev

conda env create -q -n themis-ml-ci-env python=2.7.0 \
        --file requirements_ci.txt \
        --force
source activate themis-ml-ci-env
python setup.py install
pytest
