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

"""Fine-tuning script for GemmaTE."""

import argparse
import contextlib
import os
import random
import time
import json
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from gemma import config
from gemma import model as gemma_model

# Define May 23, 2006 as the reference point (milliseconds since Unix epoch)
MAY_23_2006_MS = 1148342400000  # May 23, 2006 00:00:00 UTC in milliseconds


class TextDataset(Dataset):
    """Simple dataset for text sequences with timestamps."""
    
    def __init__(
        self, 
        texts: List[str], 
        tokenizer,
        max_length: int = 512,
        timestamps: Optional[List[int]] = None,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # If timestamps aren't provided, use current time for all texts
        if timestamps is None:
            current_time_ms = int(time.time() * 1000)
            self.timestamps = [current_time_ms] * len(texts)
        else:
            if len(timestamps) != len(texts):
                raise ValueError(f"Expected {len(texts)} timestamps, got {len(timestamps)}")
            self.timestamps = timestamps
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        timestamp = self.timestamps[idx]
        
        # Tokenize text
        tokens = self.tokenizer.encode(text)
        
        # Truncate if needed
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Create input_ids and labels tensors
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
            "timestamp": torch.tensor(timestamp, dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    
    max_length = max(x["input_ids"].size(0) for x in batch)
    
    input_ids = []
    attention_mask = []
    labels = []
    timestamps = []
    
    for x in batch:
        # Pad input_ids, attention_mask, and labels
        padding_length = max_length - x["input_ids"].size(0)
        
        input_ids.append(torch.cat([
            x["input_ids"], 
            torch.zeros(padding_length, dtype=torch.long)
        ]))
        
        attention_mask.append(torch.cat([
            x["attention_mask"], 
            torch.zeros(padding_length, dtype=torch.long)
        ]))
        
        labels.append(torch.cat([
            x["labels"], 
            torch.ones(padding_length, dtype=torch.long) * -100  # -100 is ignored by loss
        ]))
        
        timestamps.append(x["timestamp"])
    
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
        "timestamps": torch.stack(timestamps),
    }


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)


def load_dataset_from_file(data_file: str, limit: Optional[int] = None) -> Tuple[List[str], List[int]]:
    """Load dataset from a JSONL file."""
    
    texts = []
    timestamps = []
    
    print(f"Loading data from {data_file}...")
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            count = 0
            for line in f:
                example = json.loads(line.strip())
                
                # Extract prompt and response
                prompt = example.get('prompt', '')
                response = example.get('response', '')
                
                # For conversation format
                if 'conversation' in example:
                    conversation = ''
                    for turn in example['conversation']:
                        role = turn.get('role', '')
                        content = turn.get('content', '')
                        conversation += f"{role}: {content}\n"
                    text = conversation.strip()
                else:
                    # Combine prompt and response
                    text = f"{prompt}\n{response}"
                
                # Extract timestamp
                timestamp = example.get('timestamp_ms', int(time.time() * 1000))
                
                texts.append(text)
                timestamps.append(timestamp)
                
                count += 1
                if limit and count >= limit:
                    break
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Log the timestamps relative to May 23, 2006
    print(f"Loaded {len(texts)} examples from dataset")
    print("Sample data timestamps:")
    for i, ts in enumerate(timestamps[:5]):  # Show first 5 only
        timestamp_dt = datetime.fromtimestamp(ts / 1000)
        days_since_reference = (ts - MAY_23_2006_MS) / (24 * 60 * 60 * 1000)
        if days_since_reference >= 0:
            reference_str = f"{days_since_reference:.1f} days after May 23, 2006"
        else:
            reference_str = f"{abs(days_since_reference):.1f} days before May 23, 2006"
        print(f"  Sample {i}: {timestamp_dt} ({reference_str})")
    
    return texts, timestamps


def finetune(args):
    # Construct the model config
    model_config = config.get_model_config()
    model_config.dtype = "float32" if args.device == "cpu" else "bfloat16"
    model_config.quant = args.quant
    model_config.temporal_encoding_scale = args.temporal_scale
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create the model and load weights
    device = torch.device(args.device)
    with _set_default_tensor_type(model_config.get_dtype()):
        model = gemma_model.GemmaTEForCausalLM(model_config)
        
        # Load pre-trained weights if available
        if args.ckpt:
            print(f"Loading pre-trained weights from {args.ckpt}")
            model.load_weights(args.ckpt)
        
        model = model.to(device)
    
    print("GemmaTE 2b-v2 model loading complete")
    
    # Load dataset from file
    texts, timestamps = load_dataset_from_file(args.data_file, args.num_examples)
    
    # Create dataset and dataloader
    dataset = TextDataset(
        texts=texts,
        tokenizer=model.tokenizer,
        max_length=args.max_length,
        timestamps=timestamps,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    # Setup optimizer
    # Only optimize the parameters that require gradients
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(dataloader),
        eta_min=args.learning_rate / 10,
    )
    
    # Training loop
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        steps = 0
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Clear previous gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model.training_step(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                timestamps=batch["timestamps"],
            )
            
            loss = outputs["loss"]
            
            # Backward pass
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Track loss
            total_loss += loss.item()
            steps += 1
            
            if steps % args.log_interval == 0:
                avg_loss = total_loss / steps
                print(f"Step {steps}, Loss: {avg_loss:.4f}")
        
        # Epoch-level statistics
        avg_loss = total_loss / steps
        print(f"Epoch {epoch+1} complete. Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    if args.output_dir:
        final_model_path = os.path.join(args.output_dir, "gemma_te_finetuned.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
        }, final_model_path)
        print(f"Saved final model to {final_model_path}")
    
    # Sample generation with the fine-tuned model
    if args.sample_generation:
        model.eval()
        
        # Use the current time for generation
        current_time_ms = int(time.time() * 1000)
        
        # Sample prompt
        prompt = "Fine-tuning has improved my capabilities for"
        
        # Generate response
        with torch.no_grad():
            result = model.generate(
                prompt, 
                device, 
                output_len=args.sample_length,
                timestamps=torch.tensor([current_time_ms], device=device)
            )
        
        # Print sample generation
        print("\nSample generation with fine-tuned model:")
        print("----------------------------------------")
        print(f"Prompt: {prompt}")
        print(f"Result: {result}")
        print("----------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune GemmaTE model")
    
    # Model and training parameters
    parser.add_argument("--ckpt", type=str, default=None,
                       help="Path to pre-trained model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       choices=["cpu", "cuda"],
                       help="Device to run training on")
    parser.add_argument("--output_dir", type=str, default="finetuned_models",
                       help="Directory to save fine-tuned model")
    parser.add_argument("--quant", action="store_true",
                       help="Use quantized version of the model")
    parser.add_argument("--temporal_scale", type=float, default=10000.0,
                       help="Scale parameter for temporal encoding (tau)")
    
    # Dataset parameters
    parser.add_argument("--data_file", type=str, required=True,
                       help="Path to the JSONL dataset file")
    parser.add_argument("--num_examples", type=int, default=None,
                       help="Maximum number of examples to use from the dataset (None for all)")
    parser.add_argument("--max_length", type=int, default=128,
                       help="Maximum sequence length")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")
    parser.add_argument("--log_interval", type=int, default=1,
                       help="Steps between logging")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Generation parameters
    parser.add_argument("--sample_generation", action="store_true",
                       help="Generate a sample after fine-tuning")
    parser.add_argument("--sample_length", type=int, default=50,
                       help="Length of generated sample")
    
    args = parser.parse_args()
    
    finetune(args) 