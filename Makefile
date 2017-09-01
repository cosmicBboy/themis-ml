.PHONY: tests
tests:
	pytest

.PHONY: mock-ci-tests
mock-ci-tests:
	. ./ci_tests.sh
