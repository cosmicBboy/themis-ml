.PHONY: tests
tests:
	pytest

.PHONY: ci-env
ci-env:
	conda env create -q -n themis-ml-ci-env python=2.7.0 \
		--file requirements_ci.txt \
		--force && \
	source activate themis-ml-ci-env && \
	python setup.py install


.PHONY: mock-travis-ci-tests
mock-travis-ci-tests: ci-env tests
