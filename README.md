# GemmaTE: Temporal Encoding for Gemma 2b-v2

GemmaTE is an adaptation of Google's Gemma 2b-v2 model with integrated Temporal Encoding capabilities, allowing the model to understand and process time-aware information.

## Features

- **Time-Aware Text Generation**: Generate responses that are contextualized based on the current time or a specified timestamp
- **Automatic Timestamp Detection**: By default, the model uses the current timestamp for inference
- **Custom Timestamp Support**: Supply custom timestamps for historical or future-oriented text generation
- **Reference Point: May 23, 2006**: All temporal calculations use May 23, 2006 as the "zero point" for the model's understanding of time
- **Streamlined Integration**: Built on the Gemma 2b-v2 model architecture

## Installation

```bash
git clone https://github.com/yourusername/gemmaTEmporal.git
cd gemmaTEmporal
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA toolkit (recommended for GPU acceleration)

## Usage

```python
from gemma import config, model
import time

# Create model configuration and model
model_config = config.get_model_config()
model = model.GemmaTEForCausalLM(model_config)

# Generate with current timestamp (automatically uses current time)
response = model.generate("What's happening in the world right now?")
print(response)

# Generate with explicit timestamp (milliseconds since Unix epoch)
# Note: Timestamps are interpreted relative to May 23, 2006
timestamp = int(time.time() * 1000)  # current time in milliseconds
response = model.generate(
    "Tell me about recent events", 
    timestamp=timestamp
)
print(response)
```

## Scripts

### Basic Text Generation

Use the `scripts/run.py` script for basic text generation:

```bash
python scripts/run.py --prompt "What's the latest news?"
```

### Temporal Encoding Testing

Use the `scripts/run_temporal.py` script to test temporal encoding capabilities:

```bash
python scripts/run_temporal.py --prompt "What day is it today?" --temporal_scale 0.1
```

### Fine-tuning

The model provides a fine-tuning script with all necessary components:

```bash
python scripts/finetune.py --output_dir ./fine_tuned_model --num_epochs 3
```

See [Fine-tuning Guide](FINETUNING.md) for more details.

## Dataset Generation

### Single-Turn Date/Time Dataset

To generate a synthetic dataset of 100,000 examples for training the model to understand and respond to temporal queries:

```bash
python scripts/generate_temporal_dataset.py --num_examples 100000 --output_file data/temporal_dataset.jsonl
```

This script generates a variety of date/time queries and appropriate responses, with timestamps spanning from May 23, 2006 to 100 years in the future. The dataset includes:

- Date queries (e.g., "What is today's date?")
- Time queries (e.g., "What time is it?")
- Combined date-time queries
- Special date queries (holidays, etc.)
- Relative date queries ("What was the date last week?")

### Multi-Turn Conversation Dataset

For more complex training, generate multi-turn conversations focused on temporal topics:

```bash
python scripts/generate_conversation_dataset.py --num_examples 10000 --output_file data/temporal_conversations.jsonl
```

This script creates realistic conversations with multiple turns focusing on topics like:
- Basic date and time questioning
- Planning future events
- Birthday calculations
- Historical date references
- Time zone conversions
- Holiday planning
- Calendar calculations
- Season inquiries
- Age calculations
- Countdown calculations

## Using Generated Datasets for Fine-tuning

To fine-tune the model on the generated datasets:

```bash
# For single-turn examples
python scripts/finetune.py --data_file data/temporal_dataset.jsonl --output_dir fine_tuned_model

# For conversation examples
python scripts/finetune.py --data_file data/temporal_conversations.jsonl --output_dir fine_tuned_model_conv --conversation_format
```

## Temporal Reference Point

GemmaTE uses May 23, 2006, as the reference point for all temporal calculations. Timestamps are represented as milliseconds since the Unix epoch, then adjusted relative to this reference date before being encoded in the model. This specific date was chosen as the "zero point" for the model's temporal understanding.

When supplying timestamps to the model:
- For events after May 23, 2006, the value will be positive
- For events before May 23, 2006, the value will be negative

The model applies a logarithmic transformation to these relative timestamps to help handle the wide range of possible values effectively.

## Documentation

- [Temporal Encoding Details](TEMPORAL_ENCODING.md)
- [Fine-tuning Guide](FINETUNING.md)

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Acknowledgments

- GemmaTE is based on Google's Gemma model architecture.
- Temporal Encoding concept adapts and builds upon positional encoding principles from transformer architectures.
