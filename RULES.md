# Project Rules: Local LLM API

## Model Selection
- Use lightweight, CPU-friendly language models (e.g., DistilBERT, DistilGPT2, or similar small GPT variants).

## Environment Setup
- Use Python 3.10 (required for TensorFlow compatibility).
- Install TensorFlow.
- Install other necessary libraries as required.

## Model Optimization Update
- Due to current limitations, the API will use the TensorFlow (TF) model directly for inference.
- TensorFlow Lite (TFLite) conversion for DistilGPT2 is not currently feasible or supported.
- The service remains optimized for CPU by using a small model and batch processing.

## Memory Optimization Strategy
- **Standard Version**: Uses GPT-2 model (~500MB RAM) for paid hosting plans
- **Lightweight Version**: Uses tiny-gpt2 model (~100MB RAM) for free tier hosting
- Implement half-precision (float16) model loading to reduce memory usage
- Use aggressive garbage collection after model operations
- Limit batch sizes and token generation lengths based on deployment tier
- Set memory growth settings for GPU to prevent full memory allocation

## Batch Processing
- Process API inputs in small batches to manage memory usage efficiently.
- Standard version: max 3 prompts per request, 64 tokens max
- Lightweight version: max 1 prompt per request, 32 tokens max

## Frameworks
- Use standard TensorFlow for inference.
- Use FastAPI or Flask to serve the model as a web API.

## Deployment
- The API should be able to run locally as a web service.
- **Free Tier Deployment**: Use lightweight version with tiny-gpt2 model
- **Paid Tier Deployment**: Use standard version with GPT-2 model
- Configure environment variables for memory optimization (TF_FORCE_GPU_ALLOW_GROWTH, TF_CPP_MIN_LOG_LEVEL)

## API Endpoints

### 1. /generate (POST)
- Accepts a prompt (or batch of prompts) and returns generated text.
- Request JSON:
  {
    "prompts": ["string or list of strings"],
    "max_tokens": 64,
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
    "status": "ok",
    "model_loaded": true,
    "memory_usage": "optimized"
  }

## Project Structure
- main.py: FastAPI app entry point with /generate and /health endpoints (standard version)
- main_lightweight.py: FastAPI app entry point optimized for free tier deployment
- README.md: Project overview, requirements, API documentation, and usage instructions
- requirements.txt: Python dependencies
- RULES.md: Design and implementation decisions
- tasks.json: Project task tracking (to be created)
- setup.sh: Setup script for standard version
- setup_lightweight.sh: Setup script for lightweight version
- render.yaml: Render.com deployment configuration for standard version
- render_lightweight.yaml: Render.com deployment configuration for free tier

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

## TensorFlow Lite (TFLite) Status
- The API currently uses standard TensorFlow for inference, not TensorFlow Lite.
- Reason: Generative models like DistilGPT2 use layers and operations not fully supported by TensorFlow Lite. Attempts to convert the model to TFLite failed due to these limitations.
- TFLite is currently best suited for smaller, simpler models (e.g., for classification or mobile/edge use cases), not for generative LLMs.
- If TensorFlow Lite adds support for generative transformer models in the future, the API and setup scripts will be updated to take advantage of the improved performance and reduced size.

## Deployment Troubleshooting
- **Out of Memory Errors**: Switch to lightweight version or upgrade to paid plan
- **Model Loading Issues**: Verify RAM availability and internet connectivity
- **Performance Issues**: Consider reducing max_tokens or batch sizes

---
This document will be updated as new decisions and tools are added to the project. 