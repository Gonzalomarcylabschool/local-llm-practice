#!/bin/bash
set -e

# Check for Python 3.10+
PYTHON_VERSION=$(python3 -c 'import sys; print("{}.{}".format(sys.version_info[0], sys.version_info[1]))')
if [[ $(echo "$PYTHON_VERSION < 3.10" | bc) -eq 1 ]]; then
  echo "Python 3.10 or higher is required. Found $PYTHON_VERSION."
  exit 1
fi

# Install dependencies
if [ -f requirements.txt ]; then
  pip install --upgrade pip
  pip install -r requirements.txt
else
  echo "requirements.txt not found!"
  exit 1
fi

# Download DistilGPT2 model and tokenizer
python3 - <<END
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import os
os.makedirs('models/distilgpt2', exist_ok=True)
TFAutoModelForCausalLM.from_pretrained('distilgpt2', cache_dir='models/distilgpt2')
AutoTokenizer.from_pretrained('distilgpt2', cache_dir='models/distilgpt2')
print('DistilGPT2 model and tokenizer downloaded.')
END

echo "Setup complete." 