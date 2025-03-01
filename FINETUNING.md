# GemmaTE Fine-tuning Guide

This guide provides instructions for fine-tuning the GemmaTE model on your own data. GemmaTE extends the Gemma 2b-v2 model with temporal encoding capabilities, allowing it to understand time-based context.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA toolkit (recommended for GPU acceleration)
- All dependencies from requirements.txt

## Temporal Reference

GemmaTE uses May 23, 2006, as the reference point for all temporal calculations. Timestamps are represented as milliseconds since the Unix epoch, then adjusted relative to this reference date before being encoded in the model. This specific date was chosen as the "zero point" for the model's temporal understanding.

## Getting Started

1. Prepare your dataset with appropriate timestamps for each text sample
2. Run the fine-tuning script with desired hyperparameters
3. Evaluate the fine-tuned model on your specific tasks

## Basic Usage

The simplest way to fine-tune GemmaTE is to use the provided `scripts/finetune.py` script:

```bash
python scripts/finetune.py \
  --output_dir ./fine_tuned_model \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 5e-5
```

## Dataset Format

The fine-tuning script expects your data to be organized as a list of texts with corresponding timestamps. Timestamps should be in milliseconds since the Unix epoch (though they'll be adjusted relative to May 23, 2006 internally).

You can create a custom dataset by extending the `TextDataset` class provided in the script:

```python
from scripts.finetune import TextDataset

# Example with custom timestamps (milliseconds since Unix epoch)
texts = ["Sample text 1", "Sample text 2"]
timestamps = [1633046400000, 1633132800000]  # Oct 1 and Oct 2, 2021

dataset = TextDataset(texts, tokenizer, timestamps=timestamps)
```

## Advanced Configuration

The fine-tuning script supports several additional options:

- `--output_dir`: Directory to save the fine-tuned model
- `--num_epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Initial learning rate
- `--weight_decay`: Weight decay for regularization
- `--warmup_steps`: Number of warmup steps for learning rate scheduler
- `--max_length`: Maximum sequence length for training
- `--seed`: Random seed for reproducibility
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients before performing an update
- `--fp16`: Use mixed precision training (requires compatible hardware)

## Timestamp Handling

GemmaTE processes timestamps relative to May 23, 2006. The reference date, which acts as the "zero point" in the model's temporal understanding, was chosen to provide appropriate context for the model's generation capabilities.

When supplying timestamps to the model:
- For events after May 23, 2006, the value will be positive
- For events before May 23, 2006, the value will be negative

The model applies a logarithmic transformation to these relative timestamps to help handle the wide range of possible values effectively.

## Example Fine-tuning Code

Here's a more complete example of fine-tuning the model with custom data:

```python
import torch
from torch.utils.data import DataLoader
from gemma import config, model as gemma_model
from scripts.finetune import TextDataset, collate_fn

# Load model configuration and initialize model
model_config = config.get_model_config()
model = gemma_model.GemmaTEForCausalLM(model_config)

# Initialize tokenizer
tokenizer = gemma_model.Tokenizer()

# Prepare your dataset
texts = ["Your training examples here"]
timestamps = [int(time.time() * 1000)]  # Current time in milliseconds

dataset = TextDataset(texts, tokenizer, timestamps=timestamps)
dataloader = DataLoader(
    dataset, 
    batch_size=4, 
    shuffle=True, 
    collate_fn=collate_fn
)

# Set up optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
for epoch in range(3):
    for batch in dataloader:
        # Move batch to device
        batch = {k: v.to(model.device) for k, v in batch.items()}
        
        # Forward pass and calculate loss
        outputs = model.training_step(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            timestamps=batch["timestamps"],
            labels=batch["labels"]
        )
        
        loss = outputs["loss"]
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Save the fine-tuned model
torch.save(model.state_dict(), "fine_tuned_model.pt")
```

## Further Reading

For more details on GemmaTE's temporal encoding implementation, refer to the `TEMPORAL_ENCODING.md` document in this repository.

## Adapting the Model for Specific Tasks

For specific tasks like summarization or classification, you'll need to modify:

1. The training data format
2. Potentially the loss calculation in the `training_step` method
3. The evaluation metrics

## Memory Considerations

Fine-tuning large language models is memory-intensive. To fit on consumer GPUs:

1. Use gradient checkpointing (implementation pending)
2. Reduce batch size
3. Use a smaller sequence length
4. Use mixed precision training
5. Try parameter-efficient fine-tuning methods like LoRA (implementation pending)

## LoRA Adaptation (Future Work)

Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning method that can reduce memory requirements. Support for LoRA will be added in future updates.

## Saving and Loading Fine-tuned Models

The fine-tuned model is saved in PyTorch format. To load:

```python
from gemma import config, model
import torch

# Create model with default config
model_config = config.get_model_config()
model = model.GemmaTEForCausalLM(model_config)

# Load fine-tuned weights
checkpoint = torch.load("finetuned_models/gemma_te_finetuned.pt")
model.load_state_dict(checkpoint['model_state_dict'])
```

## Troubleshooting

- **Out of Memory Errors**: Reduce batch size or sequence length
- **NaN Loss**: Reduce learning rate or check for data issues
- **Slow Training**: Ensure you're using GPU acceleration
- **Poor Results**: Try adjusting hyperparameters or increasing training data

## Evaluation

To evaluate your fine-tuned model, you can:

1. Measure perplexity on a validation set
2. Test generation quality with custom prompts
3. Set up task-specific evaluation metrics 