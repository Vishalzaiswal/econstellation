from setuptools import setup, find_packages

setup(
    name="econstellation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "huggingface_hub"
    ]
)
