# Project Rules: Local LLM API

## Model Selection
- Use lightweight, CPU-friendly language models (e.g., DistilBERT, DistilGPT2, or similar small GPT variants).

## Environment Setup
- Use Python 3.10 (required for TensorFlow and TensorFlow Lite compatibility).
- Install TensorFlow and TensorFlow Lite.
- Install other necessary libraries as required.

## Model Optimization Update
- Due to current limitations, the API will use the TensorFlow (TF) model directly for inference.
- TensorFlow Lite (TFLite) conversion for DistilGPT2 is not currently feasible or supported.
- The service remains optimized for CPU by using a small model and batch processing.

## Batch Processing
- Process API inputs in small batches to manage memory usage efficiently.

## Frameworks
- Use TensorFlow Lite for optimized CPU inference.
- Use FastAPI or Flask to serve the model as a web API.

## Deployment
- The API should be able to run locally as a web service.

## API Endpoints

### 1. /generate (POST)
- Accepts a prompt (or batch of prompts) and returns generated text.
- Request JSON:
  {
    "prompts": ["string or list of strings"],
    "max_tokens": 128,
    "temperature": 1.0
  }
- Response JSON:
  {
    "results": ["generated text(s)"]
  }

### 2. /health (GET)
- Returns a simple status message to confirm the API is running.
- Response JSON:
  {
    "status": "ok"
  }

## Project Structure
- main.py: FastAPI app entry point with /generate and /health endpoints (placeholders for model logic).
- README.md: Project overview, requirements, API documentation, and usage instructions.
- requirements.txt: Python dependencies.
- RULES.md: Design and implementation decisions.
- tasks.json: Project task tracking (to be created).

## Tooling
- FastAPI for the web API.
- pyenv for Python version management (Python 3.10).
- Uvicorn for running the API server.

## Usage Clarification
- This API is designed to run as a standalone web service.
- It is intended to be accessed by other applications via HTTP requests (e.g., REST calls).
- It is not intended to be used as a direct Python library or module within other codebases.

## Repository & Setup Rules
- Do not commit model files or virtual environments to version control (see .gitignore).
- Use the provided setup script to install dependencies and download required models after cloning the repository.

---
This document will be updated as new decisions and tools are added to the project. 