# Running GemmaTE on Google Colab

This guide provides detailed instructions for setting up, fine-tuning, and running the GemmaTE model (Gemma 2b-v2 with Temporal Encoding) on Google Colab.

## Prerequisites

- A Google account to access Google Colab
- Kaggle account and API credentials (for downloading Gemma weights)
- A GPU runtime in Colab (recommended)

## Step 1: Set Up Your Colab Environment

1. Create a new Google Colab notebook: [https://colab.research.google.com](https://colab.research.google.com)
2. Set up GPU acceleration:
   - Go to **Runtime** → **Change runtime type**
   - Select **GPU** as the hardware accelerator
   - Click **Save**

## Step 2: Clone the Repository

Copy and paste this into a Colab cell, then run it:

```python
# Clone the repository
!git clone https://github.com/JamesWHomer/TemporalEncoding.git
%cd TemporalEncoding

# Note: You may need to install necessary packages depending on your environment
# The specific packages required are not listed here to give you flexibility
```

## Step 3: Create a setup.py File

The repository needs a proper setup.py file. Create one with this code in a new cell:

```python
%%writefile setup.py
from setuptools import setup, find_packages

setup(
    name="gemma",
    version="0.1",
    packages=find_packages(),
    description="GemmaTE: Temporal Encoding for Gemma 2b-v2",
)

# Install in development mode
!pip install -e .
```

## Step 4: Set Up Kaggle Credentials

You have two options for setting up Kaggle credentials:

### Option 1: Using Colab Secrets (Recommended)

1. In Colab, click on the 🔑 icon in the left sidebar to open the "Secrets" panel
2. Add your Kaggle username and API key:
   - Add a secret named `KAGGLE_USERNAME` with your username
   - Add a secret named `KAGGLE_KEY` with your API key
3. Run this code to load the credentials:

```python
import os
from google.colab import userdata

os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')
print("Kaggle credentials set from Colab secrets")
```

### Option 2: Manual Entry

If you prefer not to use Colab secrets, you can set the credentials directly:

```python
import os

os.environ["KAGGLE_USERNAME"] = "your_kaggle_username"
os.environ["KAGGLE_KEY"] = "your_kaggle_api_key"
print("Kaggle credentials set manually")
```

## Step 5: Download Model Weights

Download the pre-trained Gemma weights:

```python
# Note: You'll need to have the kagglehub package installed
import kagglehub

# Load model weights
try:
    weights_dir = kagglehub.model_download('google/gemma-2/pyTorch/gemma-2-2b-it')
    print(f"Model weights downloaded to: {weights_dir}")
except Exception as e:
    print(f"Error downloading model weights: {e}")
```

**Troubleshooting**:
- If you see authentication errors, check that your Kaggle credentials are correct
- If the download fails, you may need to accept the model license on the Kaggle website first

## Step 6: Generate the Dataset

Create a dataset for fine-tuning:

```python
# Create data directory if it doesn't exist
!mkdir -p data

# Generate temporal dataset
!python scripts/generate_temporal_dataset.py --num_examples 10000 --output_file ./data/temporal_dataset.jsonl
```

### Optional: Generate a Conversation Dataset

For multi-turn conversations:

```python
# Generate conversation dataset
!python scripts/generate_conversation_dataset.py --num_examples 5000 --output_file ./data/temporal_conversations.jsonl
```

## Step 7: Fine-tune the Model

Now you're ready to fine-tune the model:

```python
# Make sure output directory exists
!mkdir -p fine_tuned_model

# Run fine-tuning script
!python scripts/finetune.py \
  --ckpt {weights_dir} \
  --data_file ./data/temporal_dataset.jsonl \
  --output_dir ./fine_tuned_model \
  --device cuda \
  --batch_size 4 \
  --epochs 3 \
  --learning_rate 5e-5 \
  --max_length 128 \
  --sample_generation
```

**Note**: Adjust the batch size based on your GPU memory. If you encounter out-of-memory errors, try reducing the batch size to 2 or 1.

## Step 8: Test the Fine-tuned Model

After fine-tuning is complete, use this code to test the model:

```python
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
    print(f"\nPrompt: {prompt}")
    with torch.no_grad():
        result = model.generate(
            prompt, 
            device, 
            output_len=50,
            timestamps=torch.tensor([current_time_ms], device=device)
        )
    print(f"Response: {result}")
```

## Troubleshooting Common Issues

### Import Error for 'gemma' Module

If you see "No module named 'gemma'" errors:

```python
# Make sure the current directory is in the Python path
import sys
sys.path.append('.')

# Check the directory structure
!ls -la

# Verify the gemma package structure
!ls -la gemma
```

### Out of Memory Errors

If you encounter GPU memory issues:

1. Reduce the batch size (e.g., `--batch_size 1`)
2. Reduce the maximum sequence length (e.g., `--max_length 64`)
3. Use mixed precision training by adding `--dtype bfloat16` (if your GPU supports it)

### Model Loading Errors

If the model fails to load:

1. Check that the weights directory path is correct
2. Verify that all required files are present with `!ls -la {weights_dir}`

## Saving Your Work

Before ending your Colab session, save your fine-tuned model to Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy the fine-tuned model to Google Drive
!cp -r ./fine_tuned_model /content/drive/MyDrive/
```

## Using Your Model in New Sessions

To use your fine-tuned model in a new Colab session:

```python
from google.colab import drive
drive.mount('/content/drive')

# Clone the repository and set up again
!git clone https://github.com/JamesWHomer/TemporalEncoding.git
%cd TemporalEncoding
!pip install -e .

# Copy your fine-tuned model from Drive
!mkdir -p fine_tuned_model
!cp -r /content/drive/MyDrive/fine_tuned_model/* ./fine_tuned_model/

# Now you can load and use the model as shown in Step 8
``` 