# Local LLM API

A lightweight web API for running a local language model (LLM) using TensorFlow Lite, optimized for CPU inference. No calls to external APIs required.

## Features
- Runs a small LLM (e.g., DistilGPT2) locally
- Optimized for CPU with TensorFlow Lite
- FastAPI web service
- Batch processing support

## Requirements
- Python 3.10
- TensorFlow
- TensorFlow Lite
- FastAPI
- Uvicorn
- transformers

## API Endpoints

### POST /generate
- **Request:**
  ```json
  {
    "prompts": ["string or list of strings"],
    "max_tokens": 128,
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
    "status": "ok"
  }
  ```

## Usage
1. Install requirements: `pip install -r requirements.txt`
2. Start the API: `uvicorn main:app --reload`
3. Send requests to `http://localhost:8000/generate`

## Service Usage
This API is designed to run as a standalone web service. Other applications (clients) should interact with it by making HTTP requests to the API endpoints. It is not intended to be imported or used as a direct Python library.

## TensorFlow Lite (TFLite) Note
This API currently uses standard TensorFlow for DistilGPT2 inference. Attempts to convert the model to TensorFlow Lite (TFLite) were unsuccessful due to unsupported layers and operations in generative models like DistilGPT2. TFLite is best suited for smaller, simpler models (e.g., for classification or mobile/edge use cases), not for generative LLMs. If TFLite adds support for generative transformer models in the future, this API will be updated to use it for improved performance and reduced size.

---
See `RULES.md` for design and implementation decisions.