from distutils.core import setup

setup(
    name="themis-ml",
    version="0.0.1",
    description="Fairness-aware Machine Learning",
    author="Niels Bantilan",
    author_email="niels.bantilan@gmail.com",
    url="https://github.com/cosmicBboy/themis-ml",
    packages=[
        "themis_ml",
        "themis_ml.datasets",
        "themis_ml.preprocessing",
        "themis_ml.postprocessing",
        "themis_ml.linear_model"
        ],
    )
