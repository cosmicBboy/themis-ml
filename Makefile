.PHONY: tests mock-ci-tests upload-pypi clean clean_pyc conda_build_py27 \
	conda_build_py36
tests:
	pytest

mock-ci-tests:
	. ./ci_tests.sh

upload-pypi:
	python setup.py sdist upload -r pypi

clean:
	python setup.py clean

clean_pyc:
	find . -name '*.pyc' -exec rm {} \;

conda_build_py27:
	conda-build --python=2.7 conda.recipe

conda_build_py36:
	conda-build --python=3.6 conda.recipe

docs:
	@cd doc && make html
