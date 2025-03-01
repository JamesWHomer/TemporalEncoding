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

"""GemmaTE model config."""

import enum
import torch
from typing import Optional, Sequence


# Keep a mapping from dtype strings to the supported torch dtypes.
_STR_DTYPE_TO_TORCH_DTYPE = dict({
    'float16': torch.float16,
    'float': torch.float32,
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
})


class AttentionType(enum.Enum):
    GLOBAL = 1
    LOCAL_SLIDING = 2


class Architecture(enum.Enum):
    GEMMA_2 = 2


class GemmaTEConfig:
    """Configuration class for GemmaTE 2b-v2 model."""
    
    def __init__(self, dtype="bfloat16", quant=False, tokenizer="tokenizer/tokenizer.model"):
        # The architecture of the model.
        self.architecture = Architecture.GEMMA_2
        # The number of tokens in the vocabulary.
        self.vocab_size = 256000
        # The maximum sequence length that this model might ever be used with.
        self.max_position_embeddings = 8192
        # The number of blocks in the model.
        self.num_hidden_layers = 22
        # The number of attention heads used in the attention layers of the model.
        self.num_attention_heads = 8
        # The number of key-value heads for implementing attention.
        self.num_key_value_heads = 1
        # The hidden size of the model.
        self.hidden_size = 2048
        # The dimension of the MLP representations.
        self.intermediate_size = 16384
        # The number of head dimensions.
        self.head_dim = 256
        # The epsilon used by the rms normalization layers.
        self.rms_norm_eps = 1e-6
        # The dtype of the weights.
        self.dtype = dtype
        # Whether a quantized version of the model is used.
        self.quant = quant
        # The path to the model tokenizer.
        self.tokenizer = tokenizer
        # The types of attention used in the layers of the model.
        self.attn_types = [
            AttentionType.GLOBAL if i % 2 == 0 else AttentionType.LOCAL_SLIDING
            for i in range(22)
        ]
        # The size of the sliding window used for local attention.
        self.sliding_window_size = 4096
        # If provided, the final logits are softcapped to this value.
        self.final_logit_softcapping = None
        # If provided, the attention logits are softcapped to this value.
        self.attn_logit_softcapping = None
        # If provided, the query vector is normalized using the
        # inverse square root of this value instead of head_dim.
        self.query_pre_attn_scalar = None
        # Whether to use pre mlp normalization.
        self.use_pre_ffw_norm = True
        # Whether to use post mlp normalization.
        self.use_post_ffw_norm = True
        # The scale parameter for temporal encoding (tau in the formula).
        self.temporal_encoding_scale = 10000.0
        # Whether to combine temporal encoding with positional encoding.
        self.combine_temporal_and_positional = True
        # The dimension of the temporal encoding. If None, uses hidden_size.
        self.temporal_encoding_dim = None

    def get_dtype(self) -> Optional[torch.dtype]:
        """Gets the torch dtype from the config dtype string."""
        return _STR_DTYPE_TO_TORCH_DTYPE.get(self.dtype, None)


def get_model_config() -> GemmaTEConfig:
    """Returns the GemmaTE 2b-v2 model configuration."""
    return GemmaTEConfig()

