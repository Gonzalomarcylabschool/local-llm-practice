import pytest
from fastapi.testclient import TestClient
from main import app
import os

client = TestClient(app)

# Integration tests using the actual model (slow)
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_TESTS"),
    reason="Slow tests are skipped by default. Set RUN_SLOW_TESTS=1 to run."
)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_generate_single_prompt(monkeypatch):
    # Patch model and tokenizer to avoid heavy loading
    class DummyModel:
        def generate(self, input_ids, max_new_tokens, temperature, do_sample, pad_token_id):
            return [[0, 1, 2]]
    class DummyTokenizer:
        def __init__(self):
            self.eos_token_id = 0
        def encode(self, prompt, return_tensors):
            return [[0, 1]]
        def decode(self, ids, skip_special_tokens):
            return "dummy output"
    monkeypatch.setattr("main.model", DummyModel())
    monkeypatch.setattr("main.tokenizer", DummyTokenizer())
    response = client.post("/generate", json={"prompts": "hello"})
    assert response.status_code == 200
    assert response.json() == {"results": ["dummy output"]}

def test_generate_batch_prompts(monkeypatch):
    class DummyModel:
        def generate(self, input_ids, max_new_tokens, temperature, do_sample, pad_token_id):
            return [[0, 1, 2]]
    class DummyTokenizer:
        def __init__(self):
            self.eos_token_id = 0
        def encode(self, prompt, return_tensors):
            return [[0, 1]]
        def decode(self, ids, skip_special_tokens):
            return "dummy output"
    monkeypatch.setattr("main.model", DummyModel())
    monkeypatch.setattr("main.tokenizer", DummyTokenizer())
    response = client.post("/generate", json={"prompts": ["hello", "world"]})
    assert response.status_code == 200
    assert response.json() == {"results": ["dummy output", "dummy output"]}

def test_generate_model_not_loaded(monkeypatch):
    monkeypatch.setattr("main.model", None)
    monkeypatch.setattr("main.tokenizer", None)
    response = client.post("/generate", json={"prompts": "hello"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Model not loaded."

def test_generate_single_prompt_real():
    from main import startup_event
    startup_event()  # Ensure model and tokenizer are loaded
    response = client.post("/generate", json={"prompts": "Hello, my name is"})
    assert response.status_code == 200
    results = response.json()["results"]
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], str)
    assert len(results[0]) > 0

def test_generate_batch_prompts_real():
    from main import startup_event
    startup_event()  # Ensure model and tokenizer are loaded
    prompts = ["The weather today is", "Once upon a time"]
    response = client.post("/generate", json={"prompts": prompts})
    assert response.status_code == 200
    results = response.json()["results"]
    assert isinstance(results, list)
    assert len(results) == 2
    for result in results:
        assert isinstance(result, str)
        assert len(result) > 0 