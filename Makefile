.PHONY: tests
tests:
	pytest

.PHONY: mock-ci-tests
mock-ci-tests:
	. ./ci_tests.sh


.PHONY: upload-pypi
upload-pypi:
	python setup.py sdist upload -r pypi
