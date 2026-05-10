# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2/modeling_qwen2.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
from typing import Dict, Iterable, Optional, Set, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen2Config

from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, PPMissingLayer, WeightsMapper,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

logger = init_logger(__name__)


class Qwen2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen2Attention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 rope_theta: float = 10000,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 rope_scaling: Optional[Tuple] = None,
                 prefix: str = "",
                 attn_type: str = AttentionType.DECODER) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn",
                              attn_type=attn_type)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)

        # By default, Qwen2 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen2-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    })
class Qwen2Model(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 decoder_layer_type: type[nn.Module] = Qwen2DecoderLayer):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # TODO (@robertgshaw2): see if this can be moved out
        if (cache_config.sliding_window is not None
                and hasattr(config, "max_window_layers")):
            raise ValueError("Sliding window for some but all layers is not "
                             "supported. This model uses sliding window "
                             "but `max_window_layers` = {} is less than "
                             "`num_hidden_layers` = {}. Please open an issue "
                             "to discuss this feature.".format(
                                 config.max_window_layers,
                                 config.num_hidden_layers,
                             ))

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to Qwen2DecoderLayer
        decoder_layer_type = decoder_layer_type or Qwen2DecoderLayer
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: decoder_layer_type(config=config,
                                              cache_config=cache_config,
                                              quant_config=quant_config,
                                              prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen2ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen2Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              quant_config=quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)


EMBED_SIZE = 1024
HIDDEN_SIZE = 512


class SparseAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size, dtype=torch.bfloat16),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size, dtype=torch.bfloat16),
            nn.GELU(),
        )
        self.rho = 0.05
        self.rho_hat = None

    def forward(self, x):
        z = self.encoder(x)
        self.rho_hat = z.mean(dim=0)
        x_recon = self.decoder(z)
        return z, x_recon
    

class DEPModel(Qwen2ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.his_token_ids = [151665 + i for i in range(8)]
        self.diff_token_ids = [151673 + i for i in range(8)]
        # Use sets for faster lookup
        self.all_his_diff_token_ids = set(self.his_token_ids + self.diff_token_ids)
        self.his_token_id_to_idx = {tid: i for i, tid in enumerate(self.his_token_ids)}
        self.diff_token_id_to_idx = {tid: i for i, tid in enumerate(self.diff_token_ids)}
        self.sae = SparseAutoEncoder(EMBED_SIZE, HIDDEN_SIZE)
        self.align_mlp_his = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, self.config.hidden_size, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size, dtype=torch.bfloat16),
        )
        self.align_mlp_diff = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, self.config.hidden_size, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size, dtype=torch.bfloat16),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        his_diff_emb: Optional[torch.Tensor] = None,
        user_item_facets: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        inputs_embs = self.get_input_embeddings(input_ids)
        flag = False
        # Convert to Python ints first using .tolist()
        input_ids_list = input_ids.tolist()
        for tid in input_ids_list:
            if tid in self.all_his_diff_token_ids:
                flag = True
                break

        # Compute token -> batch index mapping from positions
        positions_list = positions.tolist()
        num_tokens = len(input_ids_list)
        token_batch_idx = [0] * num_tokens
        batch_idx = 0
        prev_pos = -1
        for i, p in enumerate(positions_list):
            if p < prev_pos:
                batch_idx += 1
            token_batch_idx[i] = batch_idx
            prev_pos = p

        replaced_count = 0
        if his_diff_emb is not None and flag:
            # Move to the same device as input_ids and convert to bfloat16
            his_diff_emb = his_diff_emb.to(input_ids.device).to(torch.bfloat16)

            # Handle both batched [batch, 16, 1024] and unbatched [16, 1024] cases
            if his_diff_emb.dim() == 2:
                his_diff_emb = his_diff_emb.unsqueeze(0)  # Add batch dimension

            his_diff_sparse_emb, _ = self.sae(his_diff_emb)
            his_emb = his_diff_sparse_emb[:, :8, :]
            diff_emb = his_diff_sparse_emb[:, 8:, :]
            his_emb = his_emb.to(inputs_embs.dtype)
            diff_emb = diff_emb.to(inputs_embs.dtype)
            his_emb = self.align_mlp_his(his_emb)
            diff_emb = self.align_mlp_diff(diff_emb)

            for i, tid in enumerate(input_ids_list):
                b = token_batch_idx[i]
                if tid in self.his_token_id_to_idx:
                    inputs_embs[i] = his_emb[b][self.his_token_id_to_idx[tid]]
                    replaced_count += 1
                elif tid in self.diff_token_id_to_idx:
                    inputs_embs[i] = diff_emb[b][self.diff_token_id_to_idx[tid]]
                    replaced_count += 1

        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embs)
        return hidden_states


MAX_HIS_LEN = 8
N_USER_CLUSTERS = 8
N_ITEM_CLUSTERS = 8


class FacetDEPModel(Qwen2ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Token ID layout (144 tokens total):
        # 0-63:   HIST_USER_FACET_{0-7}_{0-7} (8×8)
        # 64-127: HIST_ITEM_FACET_{0-7}_{0-7} (8×8)
        # 128-135: GLOBAL_USER_FACET_{0-7} (8)
        # 136-143: GLOBAL_ITEM_FACET_{0-7} (8)
        base_token_id = 151665

        # History user facet tokens: [HIST_USER_FACET_{i}_{j}]
        self.hist_user_facet_token_ids = []
        for i in range(MAX_HIS_LEN):
            for j in range(N_USER_CLUSTERS):
                self.hist_user_facet_token_ids.append(base_token_id + i * N_USER_CLUSTERS + j)

        # History item facet tokens: [HIST_ITEM_FACET_{i}_{j}]
        self.hist_item_facet_token_ids = []
        for i in range(MAX_HIS_LEN):
            for j in range(N_ITEM_CLUSTERS):
                self.hist_item_facet_token_ids.append(base_token_id + 64 + i * N_ITEM_CLUSTERS + j)

        # Global user facet tokens: [GLOBAL_USER_FACET_{i}]
        self.global_user_facet_token_ids = [base_token_id + 128 + i for i in range(N_USER_CLUSTERS)]

        # Global item facet tokens: [GLOBAL_ITEM_FACET_{i}]
        self.global_item_facet_token_ids = [base_token_id + 136 + i for i in range(N_ITEM_CLUSTERS)]

        # All facet tokens for quick checking - use sets and dicts for faster lookup
        self.all_facet_token_ids_set = set(
            self.hist_user_facet_token_ids +
            self.hist_item_facet_token_ids +
            self.global_user_facet_token_ids +
            self.global_item_facet_token_ids
        )

        # Create token ID to index mappings for O(1) lookup
        self.hist_user_token_id_to_idx = {tid: i for i, tid in enumerate(self.hist_user_facet_token_ids)}
        self.hist_item_token_id_to_idx = {tid: i for i, tid in enumerate(self.hist_item_facet_token_ids)}
        self.global_user_token_id_to_idx = {tid: i for i, tid in enumerate(self.global_user_facet_token_ids)}
        self.global_item_token_id_to_idx = {tid: i for i, tid in enumerate(self.global_item_facet_token_ids)}

        self.sae = SparseAutoEncoder(EMBED_SIZE, HIDDEN_SIZE)
        self.align_mlp_user = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, self.config.hidden_size, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size, dtype=torch.bfloat16),
        )
        self.align_mlp_item = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, self.config.hidden_size, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size, dtype=torch.bfloat16),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        his_diff_emb: Optional[torch.Tensor] = None,
        user_item_facets: Optional[torch.Tensor] = None,
        all_facets: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        inputs_embs = self.get_input_embeddings(input_ids)

        # Check if we need to process all_facets
        # Convert to Python ints first using .tolist() - critical fix!
        input_ids_list = input_ids.tolist()
        flag = False
        for tid in input_ids_list:
            if tid in self.all_facet_token_ids_set:
                flag = True
                break

        # Compute token -> batch index mapping from positions
        positions_list = positions.tolist()
        num_tokens = len(input_ids_list)
        token_batch_idx = [0] * num_tokens
        batch_idx = 0
        prev_pos = -1
        for i, p in enumerate(positions_list):
            if p < prev_pos:
                batch_idx += 1
            token_batch_idx[i] = batch_idx
            prev_pos = p

        replaced_count = 0
        if all_facets is not None and flag:
            # Move to the same device as input_ids and convert to bfloat16
            all_facets = all_facets.to(input_ids.device).to(torch.bfloat16)

            # Handle both batched [batch, 144, 1024] and unbatched [144, 1024] cases
            if all_facets.dim() == 2:
                all_facets = all_facets.unsqueeze(0)  # Add batch dimension

            # all_facets shape: [batch, 144, 1024]
            # Structure:
            #   0-63:   history user facets (8 positions × 8 clusters)
            #   64-127: history item facets (8 positions × 8 clusters)
            #   128-135: global user facets (8 clusters)
            #   136-143: global item facets (8 clusters)
            facets_sparse_emb, _ = self.sae(all_facets)

            # Split into different parts
            hist_user_facets_sparse = facets_sparse_emb[:, :64, :]    # [batch, 64, 512]
            hist_item_facets_sparse = facets_sparse_emb[:, 64:128, :] # [batch, 64, 512]
            global_user_facets_sparse = facets_sparse_emb[:, 128:136, :] # [batch, 8, 512]
            global_item_facets_sparse = facets_sparse_emb[:, 136:144, :] # [batch, 8, 512]

            # Reshape history facets: [batch, 8, 8, 512]
            hist_user_facets_sparse = hist_user_facets_sparse.reshape(-1, MAX_HIS_LEN, N_USER_CLUSTERS, HIDDEN_SIZE)
            hist_item_facets_sparse = hist_item_facets_sparse.reshape(-1, MAX_HIS_LEN, N_ITEM_CLUSTERS, HIDDEN_SIZE)

            # Align to LLM hidden size
            hist_user_facets_sparse = hist_user_facets_sparse.to(inputs_embs.dtype)
            hist_item_facets_sparse = hist_item_facets_sparse.to(inputs_embs.dtype)
            global_user_facets_sparse = global_user_facets_sparse.to(inputs_embs.dtype)
            global_item_facets_sparse = global_item_facets_sparse.to(inputs_embs.dtype)

            hist_user_facets_aligned = self.align_mlp_user(hist_user_facets_sparse)  # [batch, 8, 8, hidden]
            hist_item_facets_aligned = self.align_mlp_item(hist_item_facets_sparse)  # [batch, 8, 8, hidden]
            global_user_facets_aligned = self.align_mlp_user(global_user_facets_sparse)  # [batch, 8, hidden]
            global_item_facets_aligned = self.align_mlp_item(global_item_facets_sparse)  # [batch, 8, hidden]

            # Replace special tokens in input embeddings
            for i, tid in enumerate(input_ids_list):
                b = token_batch_idx[i]
                # History user facet tokens
                if tid in self.hist_user_token_id_to_idx:
                    token_idx = self.hist_user_token_id_to_idx[tid]
                    hist_pos = token_idx // N_USER_CLUSTERS
                    cluster_idx = token_idx % N_USER_CLUSTERS
                    inputs_embs[i] = hist_user_facets_aligned[b][hist_pos][cluster_idx]
                    replaced_count += 1
                # History item facet tokens
                elif tid in self.hist_item_token_id_to_idx:
                    token_idx = self.hist_item_token_id_to_idx[tid]
                    hist_pos = token_idx // N_ITEM_CLUSTERS
                    cluster_idx = token_idx % N_ITEM_CLUSTERS
                    inputs_embs[i] = hist_item_facets_aligned[b][hist_pos][cluster_idx]
                    replaced_count += 1
                # Global user facet tokens
                elif tid in self.global_user_token_id_to_idx:
                    token_idx = self.global_user_token_id_to_idx[tid]
                    inputs_embs[i] = global_user_facets_aligned[b][token_idx]
                    replaced_count += 1
                # Global item facet tokens
                elif tid in self.global_item_token_id_to_idx:
                    token_idx = self.global_item_token_id_to_idx[tid]
                    inputs_embs[i] = global_item_facets_aligned[b][token_idx]
                    replaced_count += 1

        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embs)
        return hidden_states


MAX_HIS_LEN = 8
N_BRANCHES = 8


def _make_qkv_scale_hook(q_size: int, kv_size: int):
    """Factory: create a forward hook that scales K/V at specified token positions.

    This bypasses RMSNorm (which would cancel scalar multiplication on token
    embeddings) and directly influences attention via softmax(Q·(w·K)^T/√d)
    and the weighted value contribution.
    """
    def hook(module: nn.Module, _input, output) -> torch.Tensor:
        scales: Optional[dict] = getattr(module, '_subspace_scales', None)
        if scales:
            # QKVParallelLinear returns (qkv, bias) tuple
            qkv = output[0]
            q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
            for pos, scale in scales.items():
                k[pos] = k[pos] * scale
                v[pos] = v[pos] * scale
            qkv = torch.cat([q, k, v], dim=-1)
            output = (qkv,) + output[1:]
        return output
    return hook


class SubspaceDEPModel(Qwen2ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Token IDs for subspace model
        # HIS_TOKEN_0-7: 151665-151672
        # DIFF_TOKEN_0-7: 151673-151680
        # USER_SUBSPACE_0-7: 151685-151692
        # TARGET_ITEM_SUBSPACE_0-7: 151693-151700
        # HIST_ITEM_SUBSPACE_{i}_{j}: 151701 + i*8 + j
        base = 151665

        self.his_token_ids = [base + i for i in range(8)]
        self.diff_token_ids = [base + 8 + i for i in range(8)]
        self.user_subspace_token_ids = [base + 20 + i for i in range(8)]
        self.target_item_subspace_token_ids = [base + 28 + i for i in range(8)]

        self.hist_item_subspace_token_ids = []
        for i in range(MAX_HIS_LEN):
            for j in range(N_BRANCHES):
                self.hist_item_subspace_token_ids.append(base + 36 + i * N_BRANCHES + j)

        # Sets for fast lookup
        self.all_his_diff_token_ids = set(self.his_token_ids + self.diff_token_ids)
        self.all_subspace_token_ids = set(
            self.user_subspace_token_ids +
            self.target_item_subspace_token_ids +
            self.hist_item_subspace_token_ids
        )

        # Mappings
        self.his_token_id_to_idx = {tid: i for i, tid in enumerate(self.his_token_ids)}
        self.diff_token_id_to_idx = {tid: i for i, tid in enumerate(self.diff_token_ids)}
        self.user_subspace_token_id_to_idx = {tid: i for i, tid in enumerate(self.user_subspace_token_ids)}
        self.target_item_subspace_token_id_to_idx = {tid: i for i, tid in enumerate(self.target_item_subspace_token_ids)}
        self.hist_item_subspace_token_id_to_idx = {tid: i for i, tid in enumerate(self.hist_item_subspace_token_ids)}

        # SAE for his_diff_emb (1024 -> 512)
        self.sae = SparseAutoEncoder(EMBED_SIZE, HIDDEN_SIZE)
        # SAE for subspace embeddings (512 -> 256)
        self.sae_sub = SparseAutoEncoder(HIDDEN_SIZE, HIDDEN_SIZE // 2)
        self.align_mlp_his = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, self.config.hidden_size, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size, dtype=torch.bfloat16),
        )
        self.align_mlp_diff = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, self.config.hidden_size, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size, dtype=torch.bfloat16),
        )
        # Alignment for subspace embeddings (256 -> hidden_size)
        self.align_mlp_user_sub = nn.Sequential(
            nn.Linear(HIDDEN_SIZE // 2, self.config.hidden_size, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size, dtype=torch.bfloat16),
        )
        self.align_mlp_item_sub = nn.Sequential(
            nn.Linear(HIDDEN_SIZE // 2, self.config.hidden_size, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size, dtype=torch.bfloat16),
        )

        # Register K/V scaling hook on Layer 0's qkv_proj to directly influence
        # attention distribution toward weighted subspace tokens.
        if hasattr(self.model, 'layers') and len(self.model.layers) > 0:
            layer0_attn = self.model.layers[0].self_attn
            layer0_attn.qkv_proj.register_forward_hook(
                _make_qkv_scale_hook(layer0_attn.q_size, layer0_attn.kv_size))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        his_diff_emb: Optional[torch.Tensor] = None,
        user_item_facets: Optional[torch.Tensor] = None,
        all_facets: Optional[torch.Tensor] = None,
        user_subspace_emb: Optional[torch.Tensor] = None,
        target_item_subspace_emb: Optional[torch.Tensor] = None,
        history_item_subspace_embs: Optional[torch.Tensor] = None,
        user_subspace_weights: Optional[torch.Tensor] = None,
        target_item_subspace_weights: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # Clear stale K/V scales from previous forward calls
        layer0_attn = self.model.layers[0].self_attn if hasattr(self.model, 'layers') and len(self.model.layers) > 0 else None
        if layer0_attn is not None and hasattr(layer0_attn, 'qkv_proj'):
            layer0_attn.qkv_proj._subspace_scales = None

        inputs_embs = self.get_input_embeddings(input_ids)
        input_ids_list = input_ids.tolist()

        # Compute token -> batch index mapping from positions.
        # In vLLM v0 flat batch mode, positions resets to 0 at sequence
        # boundaries, so a drop in position value signals a new sample.
        positions_list = positions.tolist()
        num_tokens = len(input_ids_list)
        token_batch_idx = [0] * num_tokens
        batch_idx = 0
        prev_pos = -1
        for i, p in enumerate(positions_list):
            if p < prev_pos:
                batch_idx += 1
            token_batch_idx[i] = batch_idx
            prev_pos = p

        # K/V scale map: token_position -> scale (float)
        kv_scales: Dict[int, float] = {}

        # Check if any special tokens are present
        flag_his_diff = False
        flag_subspace = False
        for tid in input_ids_list:
            if tid in self.all_his_diff_token_ids:
                flag_his_diff = True
            if tid in self.all_subspace_token_ids:
                flag_subspace = True
            if flag_his_diff and flag_subspace:
                break

        # Process his_diff_emb
        if his_diff_emb is not None and flag_his_diff:
            his_diff_emb = his_diff_emb.to(input_ids.device).to(torch.bfloat16)
            if his_diff_emb.dim() == 2:
                his_diff_emb = his_diff_emb.unsqueeze(0)

            his_diff_sparse_emb, _ = self.sae(his_diff_emb)
            his_emb = his_diff_sparse_emb[:, :8, :]
            diff_emb = his_diff_sparse_emb[:, 8:, :]
            his_emb = his_emb.to(inputs_embs.dtype)
            diff_emb = diff_emb.to(inputs_embs.dtype)
            his_emb = self.align_mlp_his(his_emb)
            diff_emb = self.align_mlp_diff(diff_emb)

            for i, tid in enumerate(input_ids_list):
                b = token_batch_idx[i]
                if tid in self.his_token_id_to_idx:
                    inputs_embs[i] = his_emb[b][self.his_token_id_to_idx[tid]]
                elif tid in self.diff_token_id_to_idx:
                    inputs_embs[i] = diff_emb[b][self.diff_token_id_to_idx[tid]]

        # Process user_subspace_emb with optional per-branch weights
        if user_subspace_emb is not None and flag_subspace:
            user_subspace_emb = user_subspace_emb.to(input_ids.device).to(torch.bfloat16)
            if user_subspace_emb.dim() == 2:
                user_subspace_emb = user_subspace_emb.unsqueeze(0)
            user_subspace_emb = user_subspace_emb.to(inputs_embs.dtype)
            # Pass through SAE: [N, 8, 512] -> [N*8, 512] -> [N*8, 256] -> [N, 8, 256]
            n_batch = user_subspace_emb.shape[0]
            user_subspace_emb_flat = user_subspace_emb.reshape(n_batch * N_BRANCHES, HIDDEN_SIZE)
            user_subspace_sparse, _ = self.sae_sub(user_subspace_emb_flat)
            user_subspace_emb = user_subspace_sparse.reshape(n_batch, N_BRANCHES, HIDDEN_SIZE // 2)
            user_subspace_emb = self.align_mlp_user_sub(user_subspace_emb)

            for i, tid in enumerate(input_ids_list):
                b = token_batch_idx[i]
                if tid in self.user_subspace_token_id_to_idx:
                    branch_idx = self.user_subspace_token_id_to_idx[tid]
                    inputs_embs[i] = user_subspace_emb[b][branch_idx]
                    # Record scale for K/V intervention (path 2 only)
                    if user_subspace_weights is not None:
                        kv_scales[i] = float(user_subspace_weights[b][branch_idx].item())

        # Process target_item_subspace_emb with optional per-branch weights
        if target_item_subspace_emb is not None and flag_subspace:
            target_item_subspace_emb = target_item_subspace_emb.to(input_ids.device).to(torch.bfloat16)
            if target_item_subspace_emb.dim() == 2:
                target_item_subspace_emb = target_item_subspace_emb.unsqueeze(0)
            target_item_subspace_emb = target_item_subspace_emb.to(inputs_embs.dtype)
            # Pass through SAE: [N, 8, 512] -> [N*8, 512] -> [N*8, 256] -> [N, 8, 256]
            n_batch = target_item_subspace_emb.shape[0]
            target_item_subspace_emb_flat = target_item_subspace_emb.reshape(n_batch * N_BRANCHES, HIDDEN_SIZE)
            target_item_subspace_sparse, _ = self.sae_sub(target_item_subspace_emb_flat)
            target_item_subspace_emb = target_item_subspace_sparse.reshape(n_batch, N_BRANCHES, HIDDEN_SIZE // 2)
            target_item_subspace_emb = self.align_mlp_item_sub(target_item_subspace_emb)

            for i, tid in enumerate(input_ids_list):
                b = token_batch_idx[i]
                if tid in self.target_item_subspace_token_id_to_idx:
                    branch_idx = self.target_item_subspace_token_id_to_idx[tid]
                    inputs_embs[i] = target_item_subspace_emb[b][branch_idx]
                    # Record scale for K/V intervention (path 2 only)
                    if target_item_subspace_weights is not None:
                        kv_scales[i] = float(target_item_subspace_weights[b][branch_idx].item())

        if history_item_subspace_embs is not None and flag_subspace:
            history_item_subspace_embs = history_item_subspace_embs.to(input_ids.device).to(torch.bfloat16)
            if history_item_subspace_embs.dim() == 3:
                history_item_subspace_embs = history_item_subspace_embs.unsqueeze(0)
            history_item_subspace_embs = history_item_subspace_embs.to(inputs_embs.dtype)
            # Pass through SAE: [N, 8, 8, 512] -> [N*64, 512] -> [N*64, 256] -> [N, 8, 8, 256]
            n_batch = history_item_subspace_embs.shape[0]
            history_item_subspace_emb_flat = history_item_subspace_embs.reshape(
                n_batch * MAX_HIS_LEN * N_BRANCHES, HIDDEN_SIZE
            )
            history_item_subspace_sparse, _ = self.sae_sub(history_item_subspace_emb_flat)
            history_item_subspace_embs = history_item_subspace_sparse.reshape(
                n_batch, MAX_HIS_LEN, N_BRANCHES, HIDDEN_SIZE // 2
            )
            history_item_subspace_embs = self.align_mlp_item_sub(history_item_subspace_embs)

            for i, tid in enumerate(input_ids_list):
                b = token_batch_idx[i]
                if tid in self.hist_item_subspace_token_id_to_idx:
                    token_idx = self.hist_item_subspace_token_id_to_idx[tid]
                    hist_pos = token_idx // N_BRANCHES
                    branch_idx = token_idx % N_BRANCHES
                    inputs_embs[i] = history_item_subspace_embs[b][hist_pos][branch_idx]

        # Set K/V scales on Layer 0's qkv_proj BEFORE model forward
        if layer0_attn is not None and hasattr(layer0_attn, 'qkv_proj') and kv_scales:
            layer0_attn.qkv_proj._subspace_scales = kv_scales

        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embs)
        return hidden_states

class Qwen2EmbeddingModel(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.": ""})

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        pooler_config = vllm_config.model_config.pooler_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen2Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        # TODO: Replace this model class with as_embedding_model(
        # Qwen2ForCausalLM) after changing the default pooling method
        if pooler_config.pooling_type is None:
            logger.warning(
                "This embedding model will default to last-token pooling in "
                "an upcoming version. To avoid breaking changes, you should "
                "pass `--override-pooler-config '{\"pooling_type\": \"MEAN\"}'`"
                " explicitly.")

        self._pooler = Pooler.from_config_with_defaults(
            pooler_config,
            pooling_type=PoolingType.MEAN,
            normalize=True,
            softmax=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors)

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        weights = self.hf_to_vllm_mapper.apply(weights)
        weights = ((name, data) for name, data in weights
                   if not name.startswith("lm_head."))
        self.model.load_weights(weights)
