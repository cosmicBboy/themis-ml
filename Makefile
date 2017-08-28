.PHONY: tests
tests:
	export PYTHONPATH=`git rev-parse --show-toplevel`
	pytest
