.PHONY: tests
tests:
	pytest

.PHONY: ci-tests
ci-tests:
	conda env create -q -n themis-ml-ci-env python=2.7.0 \
		--file requirements_ci.txt \
		--force && \
	. activate
	python setup.py install
	pytest
