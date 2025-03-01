# Copyright 2024 Google LLC
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
import argparse
import contextlib
import random
import time
from datetime import datetime

import numpy as np
import torch

from gemma import config
from gemma import model as gemma_model

# Define May 23, 2006 as the reference point (milliseconds since Unix epoch)
MAY_23_2006_MS = 1148342400000  # May 23, 2006 00:00:00 UTC in milliseconds

@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)


def main(args):
    # Construct the model config.
    model_config = config.get_model_config()
    model_config.dtype = "float32" if args.device == "cpu" else "float16"
    model_config.quant = args.quant
    
    # Set temporal encoding scale
    model_config.temporal_encoding_scale = args.temporal_scale
    
    # Seed random.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create the model and load the weights.
    device = torch.device(args.device)
    with _set_default_tensor_type(model_config.get_dtype()):
        model = gemma_model.GemmaTEForCausalLM(model_config)
        model.load_weights(args.ckpt)
        model = model.to(device).eval()
    print("GemmaTE 2b-v2 model loading done")
    
    # Get timestamps for demonstrations
    timestamps = []
    
    # Current time in milliseconds since epoch
    current_time_ms = int(time.time() * 1000)
    
    if args.timestamp_mode == "current":
        # Use current timestamp for all prompts
        timestamps = [current_time_ms] * len(args.prompts)
    elif args.timestamp_mode == "sequential":
        # Create a sequence of timestamps (e.g., daily intervals)
        day_ms = 24 * 60 * 60 * 1000
        base_time = current_time_ms
        timestamps = [base_time - (i * day_ms) for i in range(len(args.prompts))]
    elif args.timestamp_mode == "custom":
        # Parse custom timestamps if provided
        if args.custom_timestamps:
            timestamps = [int(ts) for ts in args.custom_timestamps.split(",")]
            if len(timestamps) != len(args.prompts):
                timestamps = timestamps * len(args.prompts)
                timestamps = timestamps[:len(args.prompts)]
        else:
            timestamps = [current_time_ms] * len(args.prompts)
    
    # Convert timestamps to tensor
    timestamps_tensor = torch.tensor(timestamps, device=device)
    
    # Generate responses for each prompt
    for i, prompt in enumerate(args.prompts):
        # For demonstration, print the timestamp in human-readable format
        timestamp_dt = datetime.fromtimestamp(timestamps[i] / 1000)
        
        # Also show time relative to May 23, 2006 reference point
        days_since_reference = (timestamps[i] - MAY_23_2006_MS) / (24 * 60 * 60 * 1000)
        if days_since_reference >= 0:
            reference_str = f"{days_since_reference:.1f} days after May 23, 2006"
        else:
            reference_str = f"{abs(days_since_reference):.1f} days before May 23, 2006"
        
        print(f"Processing prompt with timestamp: {timestamp_dt} ({reference_str})")
        
        # Generate the response
        single_timestamp = timestamps_tensor[i:i+1]
        result = model.generate(
            prompt, 
            device, 
            output_len=args.output_len,
            timestamps=single_timestamp
        )

        # Print the prompts and results
        print('======================================')
        print(f'PROMPT: {prompt}')
        print(f'TIMESTAMP: {timestamp_dt} ({reference_str})')
        print(f'RESULT: {result}')
        print('======================================')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu", 
                       choices=["cpu", "cuda"])
    parser.add_argument("--output_len", type=int, default=30)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--quant", action='store_true')
    parser.add_argument("--temporal_scale", type=float, default=10000.0,
                       help="Scale parameter for temporal encoding (tau)")
    parser.add_argument("--timestamp_mode", type=str, default="current",
                      choices=["current", "sequential", "custom"],
                      help="Mode for generating timestamps")
    parser.add_argument("--custom_timestamps", type=str, default=None,
                      help="Comma-separated list of custom timestamps (milliseconds since epoch)")
    parser.add_argument("--prompts", type=str, nargs="+", 
                      default=["What happened today?", 
                              "What were the major events from yesterday?"],
                      help="List of prompts to process with temporal encoding")
    args = parser.parse_args()

    main(args) 