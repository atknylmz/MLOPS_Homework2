from setuptools import setup, find_packages

setup(
    name="prediction-service",
    version="1.0.0",
    description="High-Cardinality Prediction Service with MLOps CI/CD Pipeline",
    author="MLOps Student",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
        "pydantic>=2.5.3",
        "numpy>=1.26.3",
        "scikit-learn>=1.4.0",
        "joblib>=1.3.2",
        "pandas>=2.1.4",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.4",
            "pytest-cov>=4.1.0",
            "httpx>=0.26.0",
            "flake8>=7.0.0",
            "pylint>=3.0.3",
            "requests>=2.31.0",
        ]
    },
)
