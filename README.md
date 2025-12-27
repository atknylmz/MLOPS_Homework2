# MLOps Homework 2 - High-Cardinality Prediction Service CI/CD Pipeline

## Project Overview

This project implements a complete MLOps CI/CD pipeline for a High-Cardinality Prediction Service. The pipeline demonstrates the transition from manual ML workflow (MLOps Level 0) to an automated pipeline (MLOps Level 1 & 2).

## Repository Structure

```
├── .github/
│   └── workflows/
│       └── main.yml              # CI/CD Pipeline Configuration
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py    # Feature hashing logic for high-cardinality features
│   ├── model.py                  # Model loading and prediction logic
│   └── api.py                    # FastAPI REST API for predictions
├── tests/
│   ├── __init__.py
│   ├── test_feature_engineering.py  # Unit tests for feature engineering
│   └── test_integration.py          # Component/Integration tests
├── scripts/
│   └── smoke_test.py            # End-to-end smoke test for deployment verification
├── data/
│   └── sample_model.joblib      # Pre-trained model artifact
├── Dockerfile                   # Container packaging
├── requirements.txt             # Python dependencies
├── .flake8                      # Flake8 linting configuration
├── setup.py                     # Package setup
└── README.md                    # This file
```

## Features

- **Feature Engineering**: Implements hash-based encoding for high-cardinality categorical features
- **REST API**: FastAPI-based prediction service
- **Comprehensive Testing**: Unit tests, integration tests, and smoke tests
- **CI/CD Pipeline**: Automated build, test, lint, package, and deploy stages
- **Docker Packaging**: Containerized deployment

## Quick Start

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run linting
flake8 src/ tests/

# Start the API locally
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build the container
docker build -t prediction-service .

# Run the container
docker run -p 8000:8000 prediction-service

# Run smoke test
python scripts/smoke_test.py
```

## CI/CD Pipeline Stages

1. **Build**: Install dependencies and prepare environment
2. **Unit Test**: Run fast, isolated tests for feature engineering logic
3. **Lint**: Static code analysis with Flake8
4. **Package**: Build Docker container (only once)
5. **Smoke Test**: Verify deployment with end-to-end health check

## Pipeline Principles

This implementation follows key MLOps principles:

- **Everything in Version Control**: All code, configurations, Dockerfiles, and test data are versioned
- **Automated Build Verification**: Every commit triggers automated testing
- **Only Build Once**: Docker image is built once and reused across stages
- **Stop the Line**: Any failure blocks deployment

## License

MIT License
