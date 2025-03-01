# Temporal Encoding in GemmaTE

This document explains the implementation and usage of temporal encoding in the GemmaTE model (Gemma 2b-v2 with temporal encoding).

## Overview

GemmaTE implements temporal encoding to provide the model with an understanding of time. This enables the model to generate time-aware responses and contextualize content based on when it was created or when the response is being generated.

## Implementation

The temporal encoding is implemented as follows:

1. Each inference call includes a timestamp (by default, the current time)
2. The timestamp is converted to a temporal embedding
3. The temporal embedding is combined with the traditional positional embeddings
4. The combined embedding is used throughout the model's layers

The temporal encoding formula is:

```
f(t) = tau * log(max(t - REF_TIME, 1))
```

Where:
- `t` is the timestamp in milliseconds
- `REF_TIME` is May 23, 2006 (our reference point) in milliseconds
- `tau` is a scaling factor (default: 0.1)
- The logarithm helps manage the scale of large timestamp values
- The max operation ensures positive values even for timestamps before the reference point

## Reference Date

GemmaTE uses May 23, 2006, as the reference point for all temporal calculations. Timestamps are represented as milliseconds since the Unix epoch (January 1, 1970), then adjusted relative to May 23, 2006 before being encoded in the model. This date serves as the "zero point" for the model's temporal understanding.

The reference date was chosen to provide a meaningful context for the model's temporal reasoning.

## Configuration

The GemmaTE model always has temporal encoding enabled by default. The configuration includes the following parameters:

- `temporal_encoding_scale` (tau): Controls the impact of temporal information
- `combine_temporal_and_positional`: Whether to add temporal encodings to positional encodings
- `temporal_encoding_dim`: Dimension of the temporal encoding vectors

## Usage

To use GemmaTE with temporal encoding:

```python
from gemma import config, model

# Create model configuration
model_config = config.get_model_config()

# Set temporal encoding scale (optional, default is 0.1)
model_config.temporal_encoding_scale = 0.1

# Create model
model = model.GemmaTEForCausalLM(model_config)

# Generate a response with the current time
response = model.generate("What is the current news?")

# Or generate a response with an explicit timestamp
# (milliseconds since Unix epoch, will be adjusted relative to May 23, 2006)
timestamp = 1640995200000  # January 1, 2022
response = model.generate("What were the major events of 2021?", timestamp=timestamp)
```

## Script Options

The `scripts/run_temporal.py` script provides a demonstration of temporal encoding capabilities:

```bash
# Run with default settings (current time)
python scripts/run_temporal.py

# Run with temporal encoding scale set to 0.5
python scripts/run_temporal.py --temporal_scale 0.5

# Run with a specific prompt
python scripts/run_temporal.py --prompt "What day is it today?"
```

## Examples

Here are examples of using the model with explicit timestamps:

```python
from gemma import config, model
import time

# Create model
model_config = config.get_model_config()
model = model.GemmaTEForCausalLM(model_config)

# Current time
current_time_ms = int(time.time() * 1000)
response = model.generate("The current time is", timestamp=current_time_ms)

# One year ago
year_ago_ms = current_time_ms - (365 * 24 * 60 * 60 * 1000)
response = model.generate("One year ago, people were talking about", timestamp=year_ago_ms)

# A specific date (January 1, 2020)
jan_2020_ms = 1577836800000
response = model.generate("In early 2020, the world was facing", timestamp=jan_2020_ms)
```

For more details on the implementation, see the `compute_temporal_encoding` function in `gemma/model.py`. 