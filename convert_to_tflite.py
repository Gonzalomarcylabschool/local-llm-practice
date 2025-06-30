import os
from transformers import TFAutoModelForCausalLM
import tensorflow as tf

MODEL_NAME = 'distilgpt2'
MODEL_DIR = 'models/distilgpt2'
SAVEDMODEL_DIR = 'models/distilgpt2_savedmodel'
TFLITE_MODEL_PATH = 'models/distilgpt2.tflite'

# Step 1: Load the model
print('Loading TensorFlow DistilGPT2 model...')
model = TFAutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR)

# Step 2: Save as SavedModel
print(f'Saving model to SavedModel format at {SAVEDMODEL_DIR}...')
model.save_pretrained(SAVEDMODEL_DIR, saved_model=True)

# Step 3: Convert to TensorFlow Lite
print('Converting SavedModel to TensorFlow Lite...')
try:
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
    tflite_model = converter.convert()
    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f'Successfully converted to TensorFlow Lite: {TFLITE_MODEL_PATH}')
except Exception as e:
    print('Error during TFLite conversion:', e) 