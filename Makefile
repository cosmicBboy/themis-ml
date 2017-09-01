.PHONY: tests
tests:
	pytest

.PHONY: ci-test
ci-test:
	conda env create -q -n themis-ml-ci-env python=2.7.0 \
		--file requirements_ci.txt \
		--force && \
	. activate
	python setup.py install
	pytest
