#!/usr/bin/env python3
# Copyright 2024
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script for fine-tuning GemmaTE on Google Colab.
This script shows the commands that should be run in a Colab notebook.
Copy each section into a separate cell in your Colab notebook.
"""

# Section 1: Clone the repository and install dependencies
"""
!git clone https://github.com/JamesWHomer/TemporalEncoding.git
%cd TemporalEncoding

# Create the setup.py file if it doesn't exist
%%writefile setup.py
from setuptools import setup, find_packages

setup(
    name="gemma",
    version="0.1",
    packages=find_packages(),
    description="GemmaTE: Temporal Encoding for Gemma 2b-v2",
)

# Install the package in development mode
!pip install -e .
"""

# Section 2: Set up Kaggle credentials for downloading the model weights
"""
# If using Kaggle to download weights, set up credentials
# You need to have added your Kaggle credentials to Colab secrets
import os
from google.colab import userdata  # Colab-specific API

# Try to get credentials from Colab secrets
try:
    os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
    os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')
    print("Kaggle credentials set from Colab secrets")
except Exception as e:
    print(f"Error setting Kaggle credentials: {e}")
    print("Please manually enter your Kaggle credentials if needed")
"""

# Section 3: Download pre-trained model weights
"""
import os
import kagglehub

# Load model weights using Kagglehub
try:
    weights_dir = kagglehub.model_download('google/gemma-2/pyTorch/gemma-2-2b-it')
    print(f"Model weights downloaded to: {weights_dir}")
except Exception as e:
    print(f"Error downloading model weights: {e}")
    # Alternative manual download instructions
    print("Alternative: Download the model from https://ai.google.dev/models/gemma")
    weights_dir = input("Enter the directory where model weights are stored: ")
"""

# Section 4: Generate the dataset
"""
# Create data directory if it doesn't exist
!mkdir -p data

# Generate temporal dataset
!python scripts/generate_temporal_dataset.py --num_examples 10000 --output_file ./data/temporal_dataset.jsonl

# Optional: Generate conversation dataset
# !python scripts/generate_conversation_dataset.py --num_examples 5000 --output_file ./data/temporal_conversations.jsonl
"""

# Section 5: Fine-tune the model
"""
# Make sure output directory exists
!mkdir -p fine_tuned_model

# Run fine-tuning script
!python scripts/finetune.py \\
  --ckpt {weights_dir} \\
  --data_file ./data/temporal_dataset.jsonl \\
  --output_dir ./fine_tuned_model \\
  --device cuda \\
  --batch_size 4 \\
  --epochs 3 \\
  --learning_rate 5e-5 \\
  --max_length 128 \\
  --sample_generation
"""

# Section 6: Test the fine-tuned model with inference
"""
import sys
sys.path.append('.')  # Ensure the current directory is in the Python path

import torch
import time
from gemma import config, model

# Create model configuration
model_config = config.get_model_config()

# Initialize the model with fine-tuned weights
model = model.GemmaTEForCausalLM(model_config)
model.load_weights("./fine_tuned_model/gemma_te_finetuned.pt")

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Generate text with current timestamp
print("Testing model with temporal queries...")
current_time_ms = int(time.time() * 1000)

# Test prompts
test_prompts = [
    "What's the date today?",
    "What day of the week is it?",
    "What time is it now?",
    "When is the next New Year's Day?"
]

for prompt in test_prompts:
    print(f"\\nPrompt: {prompt}")
    with torch.no_grad():
        result = model.generate(
            prompt, 
            device, 
            output_len=50,
            timestamps=torch.tensor([current_time_ms], device=device)
        )
    print(f"Response: {result}")
""" 