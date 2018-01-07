from setuptools import setup

setup(
    name="themis-ml",
    version="0.0.2",
    description="Fairness-aware Machine Learning",
    author="Niels Bantilan",
    author_email="niels.bantilan@gmail.com",
    url="https://github.com/cosmicBboy/themis-ml",
    download_url="https://github.com/cosmicBboy/themis-ml/archive/0.0.2.tar.gz",
    keywords=["machine-learning", "fairness-aware", "social-bias"],
    license="MIT",
    packages=[
        "themis_ml",
        "themis_ml.datasets",
        "themis_ml.linear_model",
        "themis_ml.preprocessing",
        "themis_ml.postprocessing",
        ],
    package_data={
        "themis_ml": [
            "datasets/data/german_credit.csv",
            "datasets/data/census_income_1994_1995_train.csv",
            "datasets/data/census_income_1994_1995_test.csv"
        ]
    },
    include_package_data=True,
    install_requires=[
        "scikit-learn >= 0.19.1",
        "numpy >= 1.9.0",
        "pandas >= 0.22.0",
        "pathlib2",
        ]
    )
