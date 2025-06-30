# Local LLM API

A lightweight web API for running a local language model (LLM) using standard TensorFlow, optimized for CPU inference. No calls to external APIs required.

## Features
- Runs a small LLM locally with memory optimizations
- Optimized for CPU with standard TensorFlow
- FastAPI web service
- Batch processing support
- Memory-optimized versions for different deployment scenarios

## Memory Optimization Options

### Standard Version (`main.py`)
- Uses GPT-2 model (~500MB RAM)
- Suitable for paid hosting plans (1GB+ RAM)
- Supports up to 3 prompts per request
- Max 64 tokens per generation

### Lightweight Version (`main_lightweight.py`)
- Uses tiny-gpt2 model (~100MB RAM)
- Suitable for free tier hosting (512MB RAM)
- Supports 1 prompt per request
- Max 32 tokens per generation

## Requirements
- Python 3.10
- TensorFlow
- FastAPI
- Uvicorn
- transformers

## API Endpoints

### POST /generate
- **Request:**
  ```json
  {
    "prompts": ["string or list of strings"],
    "max_tokens": 64,
    "temperature": 1.0
  }
  ```
- **Response:**
  ```json
  {
    "results": ["generated text(s)"]
  }
  ```

### GET /health
- **Response:**
  ```json
  {
    "status": "ok",
    "model_loaded": true,
    "memory_usage": "optimized"
  }
  ```

## Usage

### Local Development
1. Install requirements: `pip install -r requirements.txt`
2. Choose your version:
   - Standard: `uvicorn main:app --reload`
   - Lightweight: `uvicorn main_lightweight:app --reload`
3. Send requests to `http://localhost:8000/generate`

### Deployment
- **Free Tier (Render.com)**: Use `main_lightweight.py` with `setup_lightweight.sh`
- **Paid Tier (1GB+ RAM)**: Use `main.py` with `setup.sh`

## Memory Optimizations
- Half-precision (float16) model loading
- Aggressive garbage collection
- Limited batch sizes
- Memory growth settings for GPU
- Reduced token limits

## Service Usage
This API is designed to run as a standalone web service. Other applications (clients) should interact with it by making HTTP requests to the API endpoints. It is not intended to be imported or used as a direct Python library.

## TensorFlow Lite (TFLite) Note
This API currently uses standard TensorFlow for inference. Attempts to convert the model to TensorFlow Lite (TFLite) were unsuccessful due to unsupported layers and operations in generative models. TFLite is best suited for smaller, simpler models (e.g., for classification or mobile/edge use cases), not for generative LLMs. If TFLite adds support for generative transformer models in the future, this API will be updated to use it for improved performance and reduced size.

## Deployment Troubleshooting

### Out of Memory Errors
If you encounter OOM errors on Render.com:
1. Use the lightweight version (`main_lightweight.py`)
2. Upgrade to a paid plan with more RAM
3. Reduce `max_tokens` in requests
4. Limit concurrent requests

### Model Loading Issues
- Ensure sufficient RAM for your chosen model
- Check internet connectivity for model download
- Verify Python 3.10 compatibility

---
See `RULES.md` for design and implementation decisions.