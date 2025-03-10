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
"""Inference-only GemmaTE model implementation."""

import json
import gc
import os
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, List, Optional, Sequence, Tuple, Union
import time

from gemma import config as gemma_config
from gemma import tokenizer


class Sampler(nn.Module):

    def __init__(self, vocab_size: int, config: gemma_config.GemmaTEConfig):
        super().__init__()
        self.vocab_size = vocab_size
        self.config = config

    @torch.no_grad()
    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Select the last element for each sequence.
        # (batch_size, input_len, hidden_size) -> (batch_size, hidden_size)
        hidden_states = hidden_states.index_select(
            1, output_positions).squeeze(dim=1)
            
        # Ensure hidden_states and embedding have the same dtype before matmul
        if hidden_states.dtype != embedding.dtype:
            hidden_states = hidden_states.to(embedding.dtype)
            
        logits = torch.matmul(hidden_states, embedding.t())
        if embedding_bias is not None:
            logits += embedding_bias
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        if temperatures is None:
            return torch.argmax(logits, dim=-1).squeeze(dim=-1), logits

        # Apply temperature scaling.
        logits.div_(temperatures.unsqueeze(dim=1))

        # Calculate probabilities with softmax.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        # Apply top-p, top-k.
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_ps_mask = (probs_sum - probs_sort) > top_ps.unsqueeze(dim=1)
        probs_sort = torch.where(top_ps_mask, 0, probs_sort)

        top_ks_mask = torch.arange(probs_idx.shape[-1],
                                   device=probs_idx.device)
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
        top_ks_mask = top_ks_mask >= top_ks.unsqueeze(dim=1)
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)

        # Re-normalization.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        probs = torch.gather(probs_sort,
                             dim=-1,
                             index=torch.argsort(probs_idx, dim=-1))

        next_token_ids = torch.multinomial(probs,
                                           num_samples=1,
                                           replacement=True).squeeze(dim=-1)
        return next_token_ids, logits


def precompute_freqs_cis(dim: int,
                         end: int,
                         theta: float = 10000.0) -> torch.Tensor:
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def compute_temporal_encoding(timestamps: torch.Tensor, 
                              dim: int,
                              theta: float = 10000.0) -> torch.Tensor:
    """Computes temporal encoding for a batch of timestamps.
    
    Args:
        timestamps: A tensor of timestamps [batch_size] in milliseconds since Unix epoch.
        dim: The dimension of the temporal encoding.
        theta: The scale parameter.
        
    Returns:
        A tensor of temporal encodings [batch_size, dim].
    """
    # Define May 23, 2006 as the reference point (milliseconds since Unix epoch)
    MAY_23_2006_MS = 1148342400000  # May 23, 2006 00:00:00 UTC in milliseconds

    # Adjust timestamps to be relative to May 23, 2006
    relative_timestamps = timestamps.float() - MAY_23_2006_MS
    
    # Handle cases where timestamps are before the reference point
    relative_timestamps = torch.sign(relative_timestamps) * torch.log(torch.abs(relative_timestamps) + 1)
    
    # Create sinusoidal embeddings similar to positional embeddings
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    freqs = freqs.to(timestamps.device)
    
    # [batch_size, dim//2]
    embeddings = relative_timestamps.unsqueeze(1) * freqs.unsqueeze(0)
    
    # [batch_size, dim]
    embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=1)
    
    # If dim is odd, pad with zeros
    if dim % 2 == 1:
        embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=1)
        
    return embeddings


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    # Make sure we're working with float tensors for the complex operations
    x_float = x.transpose(1, 2).float()
    
    # Split the head dimension into real and imaginary parts
    x_chunked = torch.chunk(x_float, 2, dim=-1)
    x_complex = torch.view_as_complex(torch.stack(x_chunked, dim=-1))
    
    # Handle batched vs. unbatched freqs_cis
    if freqs_cis.dim() == 1 and x_complex.dim() > 1:
        # For training, we need to expand freqs_cis to match the sequence length
        seq_length = x_complex.size(-2)
        if freqs_cis.size(0) < seq_length:
            # If we need more positions than we have in freqs_cis, something is wrong
            raise ValueError(f"freqs_cis has fewer positions ({freqs_cis.size(0)}) than required ({seq_length})")
        # Just use the first seq_length positions from freqs_cis
        freqs_cis = freqs_cis[:seq_length]
        
    # Apply the rotation in complex space
    x_out = torch.view_as_real(x_complex * freqs_cis.unsqueeze(0).unsqueeze(0))
    
    # Convert back to the original format
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1)
    
    # Return with the same dtype as the input
    return x_out.transpose(1, 2).type_as(x)


class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, quant: bool):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features), dtype=torch.int8),
                requires_grad=False,
            )
            self.weight_scaler = nn.Parameter(torch.Tensor(out_features))
        else:
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features)),
                requires_grad=False,
            )
        self.quant = quant

    def forward(self, x):
        weight = self.weight
        if self.quant:
            weight = weight * self.weight_scaler.unsqueeze(-1)
        # Ensure input x has the same dtype as weight
        if x.dtype != weight.dtype:
            x = x.to(weight.dtype)
        output = F.linear(x, weight)
        return output


class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, quant: bool):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), dtype=torch.int8),
                requires_grad=False,
            )
            self.weight_scaler = nn.Parameter(torch.Tensor(num_embeddings))
        else:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim)),
                requires_grad=False,
            )
        self.quant = quant

    def forward(self, x):
        weight = self.weight
        if self.quant:
            weight = weight * self.weight_scaler.unsqueeze(-1)
        # Ensure input x has compatible dtype with weight for embedding
        if x.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
            x = x.to(torch.int64)  # Convert to long for embedding indices
        output = F.embedding(x, weight)
        return output


class RMSNorm(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = self._norm(x.float())
        if self.add_unit_offset:
            output = output * (1 + self.weight.float())
        else:
            output = output * self.weight.float()
        return output.type_as(x)


class GemmaTEMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant: bool,
    ):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, quant)
        self.up_proj = Linear(hidden_size, intermediate_size, quant)
        self.down_proj = Linear(intermediate_size, hidden_size, quant)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs


class GemmaTEAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        attn_logit_softcapping: Optional[float],
        query_pre_attn_scalar: Optional[int],
        head_dim: int,
        quant: bool,
        attn_type: gemma_config.AttentionType,
        sliding_window_size: Optional[int] = None,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = hidden_size
        self.head_dim = head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        if query_pre_attn_scalar is not None:
            self.scaling = query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5

        self.qkv_proj = Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            quant=quant)
        self.o_proj = Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            quant=quant)

        self.attn_type = attn_type
        self.sliding_window_size = sliding_window_size
        self.attn_logit_softcapping = attn_logit_softcapping

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3

        batch_size, input_len, _ = hidden_states_shape

        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size],
                               dim=-1)

        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        # Positional embedding - ensure freqs_cis has the right shape
        if freqs_cis.dim() == 1:
            # During training, freqs_cis is a 1D tensor of input_len positions
            # Make sure we have enough positions
            if freqs_cis.size(0) < input_len:
                raise ValueError(f"freqs_cis has too few positions ({freqs_cis.size(0)}) for input length {input_len}")
            # Only use the positions we need
            freqs_cis = freqs_cis[:input_len]
        
        # Apply rotary embeddings
        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)

        # Write new kv cache.
        # [batch_size, input_len, n_local_kv_heads, head_dim]
        k_cache, v_cache = kv_cache
        k_cache.index_copy_(1, kv_write_indices, xk)
        v_cache.index_copy_(1, kv_write_indices, xv)

        key = k_cache
        value = v_cache
        if self.num_kv_heads != self.num_heads:
            # [batch_size, max_seq_len, n_local_heads, head_dim]
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(value,
                                            self.num_queries_per_kv,
                                            dim=2)

        # [batch_size, n_local_heads, input_len, head_dim]
        q = xq.transpose(1, 2)
        # [batch_size, n_local_heads, max_seq_len, head_dim]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        # [batch_size, n_local_heads, input_len, max_seq_len]
        q.mul_(self.scaling)
        scores = torch.matmul(q, k.transpose(2, 3))
        if (
            self.attn_type == gemma_config.AttentionType.LOCAL_SLIDING
            and self.sliding_window_size is not None
        ):
            all_ones = torch.ones_like(mask)
            sliding_mask = torch.triu(
                all_ones, -1 * self.sliding_window_size + 1
            ) * torch.tril(all_ones, self.sliding_window_size - 1)
            mask = torch.where(sliding_mask == 1, mask, -2.3819763e38)
        if self.attn_logit_softcapping is not None:
            scores = scores / self.attn_logit_softcapping
            scores = torch.tanh(scores)
            scores = scores * self.attn_logit_softcapping
        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)

        # [batch_size, n_local_heads, input_len, head_dim]
        output = torch.matmul(scores, v)

        # [batch_size, input_len, hidden_dim]
        output = (output.transpose(1, 2).contiguous().view(
            batch_size, input_len, -1))
        output = self.o_proj(output)
        return output


class GemmaTEDecoderLayer(nn.Module):

    def __init__(
        self,
        config: gemma_config.GemmaTEConfig,
    ):
        super().__init__()
        self.self_attn = GemmaTEAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            attn_logit_softcapping=config.attn_logit_softcapping,
            query_pre_attn_scalar=config.query_pre_attn_scalar,
            head_dim=config.head_dim,
            quant=config.quant,
            attn_type=gemma_config.AttentionType.GLOBAL,
        )
        self.mlp = GemmaTEMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=config.quant,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GemmaTE2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: gemma_config.GemmaTEConfig,
        attn_type: gemma_config.AttentionType,
    ):
        super().__init__()
        self.self_attn = GemmaTEAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            attn_logit_softcapping=config.attn_logit_softcapping,
            query_pre_attn_scalar=config.query_pre_attn_scalar,
            head_dim=config.head_dim,
            quant=config.quant,
            attn_type=attn_type,
            sliding_window_size=config.sliding_window_size,
        )
        self.mlp = GemmaTEMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=config.quant,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_pre_ffw_norm
            else None
        )
        self.post_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_post_ffw_norm
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        if self.pre_feedforward_layernorm is not None:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.post_feedforward_layernorm is not None:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GemmaTEModel(nn.Module):

    def __init__(self, config: gemma_config.GemmaTEConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            # All models now use GEMMA_2 architecture
            attn_type = (
                config.attn_types[i]
                if config.attn_types is not None
                else gemma_config.AttentionType.GLOBAL
            )
            self.layers.append(GemmaTE2DecoderLayer(config, attn_type))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                kv_write_indices=kv_write_indices,
                kv_cache=kv_caches[i],
                mask=mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class GemmaTEForCausalLM(nn.Module):

    def __init__(
        self,
        config: gemma_config.GemmaTEConfig,
    ):
        super().__init__()
        self.config = config
        assert config.hidden_size % config.num_attention_heads == 0

        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim
        vocab_size = config.vocab_size

        self.tokenizer = tokenizer.Tokenizer(config.tokenizer)
        self.embedder = Embedding(vocab_size, config.hidden_size, config.quant)
        self.model = GemmaTEModel(config)
        self.sampler = Sampler(vocab_size, config)

        # Pre-compute rotary embedding table.
        rope_theta = getattr(config, 'rope_theta', 10000)
        freqs_cis = precompute_freqs_cis(head_dim,
                                         max_seq_len * 2,
                                         theta=rope_theta)
        self.register_buffer('freqs_cis', freqs_cis)
        
        # Setup for temporal encoding (always enabled)
        self.temporal_encoding_scale = config.temporal_encoding_scale
        self.temporal_encoding_dim = config.temporal_encoding_dim or config.hidden_size
        # Linear projection for temporal embeddings if dimensions don't match
        if self.temporal_encoding_dim != config.hidden_size:
            self.temporal_proj = Linear(self.temporal_encoding_dim, 
                                       config.hidden_size,
                                       quant=config.quant)

    @torch.no_grad()
    def forward(
        self,
        input_token_ids: torch.Tensor,
        input_positions: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs_cis = self.freqs_cis.index_select(0, input_positions)
        kv_write_indices = input_positions

        # [batch_size, input_len, hidden_size]
        hidden_states = self.embedder(input_token_ids)
        # Gemma normalizes the embedding by sqrt(hidden_size).
        # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer
        
        # Apply temporal encoding if timestamps are provided
        if timestamps is not None:
            batch_size, seq_len, _ = hidden_states.shape
            
            # Compute temporal encoding for each timestamp
            temporal_embedding = compute_temporal_encoding(
                timestamps, 
                self.temporal_encoding_dim,
                self.temporal_encoding_scale
            )
            
            # Project to hidden_size if dimensions don't match
            if hasattr(self, 'temporal_proj'):
                temporal_embedding = self.temporal_proj(temporal_embedding)
            
            # Reshape to match hidden_states and add
            # [batch_size, 1, hidden_size]
            temporal_embedding = temporal_embedding.unsqueeze(1)
            # [batch_size, seq_len, hidden_size]
            temporal_embedding = temporal_embedding.expand(-1, seq_len, -1)
            
            # Add temporal embeddings to token embeddings
            hidden_states = hidden_states + temporal_embedding

        hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_caches=kv_caches,
            mask=mask,
        )
        embedder_weight = self.embedder.weight
        if self.config.quant:
            embedder_weight = (
                embedder_weight * self.embedder.weight_scaler.unsqueeze(-1))
        next_tokens, logits = self.sampler(
            embedding=embedder_weight,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )
        return next_tokens, logits

    def generate(
        self,
        prompts: Union[str, Sequence[str]],
        device: Any,
        output_len: int = 100,
        temperature: Union[float, None] = 0.95,
        top_p: float = 1.0,
        top_k: int = 100,
        timestamps: Optional[Union[torch.Tensor, List[int]]] = None,
    ) -> Union[str, Sequence[str]]:
        """Generates responses for given prompts using Gemma model."""
        # If a single prompt is provided, treat it as a batch of 1.
        is_str_prompt = isinstance(prompts, str)
        if is_str_prompt:
            prompts = [prompts]

        batch_size = len(prompts)
        
        # Get the model's configured dtype
        dtype = self.config.get_dtype()
        
        # Process timestamps if provided
        if timestamps is not None:
            if isinstance(timestamps, list):
                timestamps = torch.tensor(timestamps, device=device)
            elif not isinstance(timestamps, torch.Tensor):
                raise ValueError("Timestamps must be provided as a list of integers or a torch.Tensor")
            
            # If a single timestamp is provided for multiple prompts, expand it
            if timestamps.numel() == 1 and batch_size > 1:
                timestamps = timestamps.expand(batch_size)
                
            # Ensure timestamps tensor has the correct shape [batch_size]
            if timestamps.shape[0] != batch_size:
                raise ValueError(f"Expected timestamps for {batch_size} prompts, got {timestamps.shape[0]}")
            
            # Ensure timestamps has the right dtype
            if timestamps.dtype != torch.long:
                timestamps = timestamps.to(torch.long)
        else:
            # If no timestamps provided, use current time (milliseconds since epoch)
            current_time_ms = int(time.time() * 1000)
            timestamps = torch.tensor([current_time_ms] * batch_size, device=device, dtype=torch.long)
            
        prompt_tokens = [self.tokenizer.encode(prompt) for prompt in prompts]
        min_prompt_len = min(len(p) for p in prompt_tokens)
        max_prompt_len = max(len(p) for p in prompt_tokens)
        max_seq_len = max_prompt_len + output_len
        assert max_seq_len <= self.config.max_position_embeddings

        # build KV caches
        kv_caches = []
        for _ in range(self.config.num_hidden_layers):
            size = (batch_size, max_seq_len, self.config.num_key_value_heads,
                    self.config.head_dim)
            k_cache = torch.zeros(size=size, dtype=dtype, device=device)
            v_cache = torch.zeros(size=size, dtype=dtype, device=device)
            kv_caches.append((k_cache, v_cache))

        # prepare inputs
        token_ids_tensor = torch.full((batch_size, max_seq_len),
                                      self.tokenizer.pad_id, dtype=torch.int64)
        input_token_ids_tensor = torch.full((batch_size, min_prompt_len),
                                            self.tokenizer.pad_id,
                                            dtype=torch.int64)
        for i, p in enumerate(prompt_tokens):
            token_ids_tensor[i, :len(p)] = torch.tensor(p)
            input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(
                p[:min_prompt_len])
        token_ids_tensor = token_ids_tensor.to(device)
        input_token_ids_tensor = input_token_ids_tensor.to(device)
        prompt_mask_tensor = token_ids_tensor != self.tokenizer.pad_id
        input_positions_tensor = torch.arange(0, min_prompt_len,
                                              dtype=torch.int64).to(device)
        mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len),
                                 -2.3819763e38, dtype=dtype).to(device)
        mask_tensor = torch.triu(mask_tensor, diagonal=1)
        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(
            device)
        temperatures_tensor = None if not temperature else torch.FloatTensor(
            [temperature] * batch_size).to(device).to(dtype)
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device).to(dtype)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        output_index = torch.tensor(min_prompt_len, dtype=torch.int64).to(
            device)

        # Prefill up to min_prompt_len tokens, then treat other prefill as
        # decode and ignore output.
        for i in range(max_seq_len - min_prompt_len):
            next_token_ids, _ = self(
                input_token_ids=input_token_ids_tensor,
                input_positions=input_positions_tensor,
                kv_write_indices=None,
                kv_caches=kv_caches,
                mask=curr_mask_tensor,
                output_positions=output_positions_tensor,
                temperatures=temperatures_tensor,
                top_ps=top_ps_tensor,
                top_ks=top_ks_tensor,
                timestamps=timestamps,
            )

            curr_prompt_mask = prompt_mask_tensor.index_select(
                1, output_index).squeeze(dim=1)
            curr_token_ids = token_ids_tensor.index_select(
                1, output_index).squeeze(dim=1)
            output_token_ids = torch.where(curr_prompt_mask, curr_token_ids,
                                           next_token_ids).unsqueeze(dim=1)
            token_ids_tensor.index_copy_(1, output_index, output_token_ids)

            input_token_ids_tensor = output_token_ids
            input_positions_tensor = output_index.unsqueeze(dim=-1)
            curr_mask_tensor = mask_tensor.index_select(2,
                                                        input_positions_tensor)
            output_positions_tensor = torch.tensor(0, dtype=torch.int64).to(
                device)
            output_index = output_index + 1

        # Detokenization.
        token_ids = token_ids_tensor.tolist()
        results = []
        for i, tokens in enumerate(token_ids):
            trimmed_output = tokens[len(prompt_tokens[i]):len(prompt_tokens[i])
                                    + output_len]
            if self.tokenizer.eos_id in trimmed_output:
                eos_index = trimmed_output.index(self.tokenizer.eos_id)
                trimmed_output = trimmed_output[:eos_index]
            results.append(self.tokenizer.decode(trimmed_output))

        # If a string was provided as input, return a string as output.
        return results[0] if is_str_prompt else results

    def load_weights(self, model_path: str):
        # Case 1: Direct model file (e.g., finetuned checkpoint)
        if os.path.isfile(model_path):
            print(f"Loading weights from direct file: {model_path}")
            self.load_state_dict(
                torch.load(
                    model_path, mmap=True, weights_only=True,
                )['model_state_dict'],
                strict=False,
            )
        # Case 2: Check if model.ckpt exists directly in the given path (when path already ends with /1)
        elif os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, 'model.ckpt')):
            ckpt_path = os.path.join(model_path, 'model.ckpt')
            print(f"Loading weights from checkpoint: {ckpt_path}")
            self.load_state_dict(
                torch.load(
                    ckpt_path, mmap=True, weights_only=True,
                ),
                strict=False,
            )
        # Case 3: Check for directory structure like gemma-2-2b-it/1/model.ckpt
        elif os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, '1', 'model.ckpt')):
            ckpt_path = os.path.join(model_path, '1', 'model.ckpt')
            print(f"Loading weights from checkpoint: {ckpt_path}")
            self.load_state_dict(
                torch.load(
                    ckpt_path, mmap=True, weights_only=True,
                ),
                strict=False,
            )
        # Case 4: HuggingFace-style sharded weights
        elif os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, 'pytorch_model.bin.index.json')):
            print(f"Loading weights from sharded files in: {model_path}")
            index_path = os.path.join(model_path, 'pytorch_model.bin.index.json')
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            shard_files = list(set(index["weight_map"].values()))
            for shard_file in shard_files:
                shard_path = os.path.join(model_path, shard_file)
                state_dict = torch.load(shard_path, map_location="cpu", weights_only=True)
                self.load_state_dict(state_dict, strict=False)
                del state_dict  # Save memory.
                gc.collect()
        else:
            raise ValueError(f"Could not find model weights at {model_path}. Expected either a direct file, "
                            f"a directory with 'model.ckpt', a directory with '1/model.ckpt', "
                            f"or a directory with 'pytorch_model.bin.index.json'.")

    def training_step(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None
    ) -> dict:
        """Performs a training step and returns the loss.
        
        Args:
            input_ids: Token ids of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len] 
            labels: Target token ids of shape [batch_size, seq_len]
            timestamps: Optional timestamps of shape [batch_size]
            
        Returns:
            Dictionary containing the loss and logits
        """
        dtype = self.config.get_dtype()
        batch_size, seq_len = input_ids.shape
        
        # Create positions tensor for each position in the sequence
        positions = torch.arange(0, seq_len, device=input_ids.device)
        
        # Get rotary embeddings - similar to how it's done in forward method
        freqs_cis = self.freqs_cis.index_select(0, positions).to(input_ids.device)
        
        # Get embeddings with input ids
        hidden_states = self.embedder(input_ids)
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=dtype)
        hidden_states = hidden_states * normalizer
        
        # Apply temporal encoding if timestamps are provided
        if timestamps is not None:
            # Ensure timestamps has the right dtype
            if timestamps.dtype != torch.long:
                timestamps = timestamps.to(torch.long)
                
            temporal_embedding = compute_temporal_encoding(
                timestamps, 
                self.temporal_encoding_dim,
                self.temporal_encoding_scale
            )
            
            # Project to hidden_size if dimensions don't match
            if hasattr(self, 'temporal_proj'):
                temporal_embedding = self.temporal_proj(temporal_embedding)
                
            # Reshape to match hidden_states and add
            temporal_embedding = temporal_embedding.unsqueeze(1)
            temporal_embedding = temporal_embedding.expand(-1, seq_len, -1)
            
            # Ensure consistent dtype
            if temporal_embedding.dtype != dtype:
                temporal_embedding = temporal_embedding.to(dtype)
                
            # Add temporal embeddings to token embeddings
            hidden_states = hidden_states + temporal_embedding
            
        # Ensure hidden_states has the correct dtype
        if hidden_states.dtype != dtype:
            hidden_states = hidden_states.to(dtype)
            
        # Create causal attention mask with the right shape for training
        mask = torch.triu(torch.full((1, 1, seq_len, seq_len), -float('inf'), device=input_ids.device), diagonal=1)
        
        # Apply attention mask from padding if provided
        if attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Convert to the format expected by the model
            attention_mask = (1.0 - attention_mask) * -1e9
            mask = mask + attention_mask
        
        # Ensure mask has the right dtype
        if mask.dtype != dtype and mask.dtype != torch.bool:
            mask = mask.to(dtype)
            
        # Build temporary KV caches for training - flatten positions for index_copy_
        kv_write_indices = torch.arange(seq_len, device=input_ids.device)
        
        # Create KV caches similar to the forward pass
        kv_caches = []
        for _ in range(self.config.num_hidden_layers):
            size = (batch_size, seq_len, self.config.num_key_value_heads, self.config.head_dim)
            k_cache = torch.zeros(size=size, dtype=dtype, device=input_ids.device)
            v_cache = torch.zeros(size=size, dtype=dtype, device=input_ids.device)
            kv_caches.append((k_cache, v_cache))
            
        # Forward pass through model
        hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_caches=kv_caches,
            mask=mask,
        )
        
        # Ensure hidden_states has the correct dtype after model forward pass
        if hidden_states.dtype != dtype:
            hidden_states = hidden_states.to(dtype)
        
        # Get logits
        embedder_weight = self.embedder.weight
        if self.config.quant:
            embedder_weight = embedder_weight * self.embedder.weight_scaler.unsqueeze(-1)
            
        # Ensure embedder_weight has the correct dtype
        if embedder_weight.dtype != dtype:
            embedder_weight = embedder_weight.to(dtype)
            
        logits = torch.matmul(hidden_states, embedder_weight.t())
        
        # Apply logit softcapping if configured
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
            
        # Calculate loss
        loss = None
        if labels is not None:
            # Shift the logits and labels for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            # Only calculate loss on non-padding tokens
            active_mask = shift_labels != self.tokenizer.pad_id
            if active_mask.sum() > 0:
                active_logits = shift_logits[active_mask]
                active_labels = shift_labels[active_mask]
                
                # Always use float32 for loss calculation (needed by CrossEntropyLoss)
                if active_logits.dtype != torch.float32:
                    active_logits = active_logits.float()
                    
                loss = loss_fct(active_logits, active_labels)
            else:
                # Create a zero loss tensor with the appropriate device
                loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
                
        return {
            "loss": loss,
            "logits": logits
        }
