#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
#
"""Quantization configuration classes for Ascend."""

from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, cast

import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import (
    QUANTIZATION_METHODS, register_quantization_config)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import \
    CompressedTensorsScheme
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    find_matched_target, is_activation_quantization_format,
    should_ignore_layer)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod, VocabParallelEmbedding)
from vllm.model_executor.models.utils import WeightsMapper
from vllm.logger import init_logger
from compressed_tensors.quantization import QuantizationStrategy, QuantizationArgs

from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD, COMPRESSED_TENSORS_METHOD

from .model_mappings import QUANT_MODEL_PREFIX_MAPPINGS, packed_modules_model_mapping
from .wrappers import (AscendEmbeddingMethod, AscendFusedMoEMethod,
                       AscendKVCacheMethod, AscendLinearMethod)


logger = init_logger(__name__)


@register_quantization_config(ASCEND_QUANTIZATION_METHOD)
class AscendQuantConfig(QuantizationConfig):
    """Config class for Ascend ModelSlim quantization.

    This class is a general class that parses quantization configs
    that are supported on Ascend hardware.
    """

    def __init__(self, quant_config: Dict[str, Any]):
        super().__init__()
        self.quant_description = quant_config
        # TODO(whx): remove this adaptation after adding "shared_head"
        # to prefix of DeepSeekShareHead in vLLM.
        extra_quant_dict = {}
        for k in self.quant_description.keys():
            if "shared_head" in k:
                new_k = k.replace(".shared_head.", ".")
                extra_quant_dict[new_k] = self.quant_description[k]
            if "weight_packed" in k:
                new_k = k.replace("weight_packed", "weight")
                extra_quant_dict[new_k] = self.quant_description[k]
        self.quant_description.update(extra_quant_dict)

    def __repr__(self) -> str:
        return "AscendQuantConfig:\n" + super().__repr__()

    @classmethod
    def get_name(cls) -> str:
        return ASCEND_QUANTIZATION_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.int8, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            "Ascend hardware dose not support \"get_min_capability\" feature.")

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quant_model_description.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AscendQuantConfig":
        return cls(config)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        if hf_quant_cfg is not None:
            quant_method = hf_quant_cfg.get("quant_method", None)
            if not quant_method and torch.npu.is_available():
                return ASCEND_QUANTIZATION_METHOD
        return None

    def quant_prefix_mapper(self, model_type: str, prefix: str) -> str:
        # TODO (Levi-JQ): will be removed when QuantizationConfig.apply_vllm_mapper is implemented
        prefix_mapping = QUANT_MODEL_PREFIX_MAPPINGS.get(model_type)
        if prefix_mapping:
            hf_to_vllm_mapper = WeightsMapper(
                orig_to_new_prefix=prefix_mapping)
            return hf_to_vllm_mapper._map_name(prefix)
        return prefix

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        vllm_config = get_current_vllm_config()
        model_type = vllm_config.model_config.hf_config.model_type
        if model_type in packed_modules_model_mapping:
            self.packed_modules_mapping = packed_modules_model_mapping[
                model_type]
        prefix = self.quant_prefix_mapper(model_type, prefix)
        from vllm.attention.layer import Attention
        if prefix.startswith("language_model"):
            prefix = prefix.split('.', 1)[-1]
        if isinstance(layer, LinearBase):
            if self.is_layer_skipped_ascend(prefix,
                                            self.packed_modules_mapping):
                return AscendUnquantizedLinearMethod()
            return AscendLinearMethod(self, prefix,
                                      self.packed_modules_mapping, layer)
        elif isinstance(layer, Attention) and \
            'fa_quant_type' in self.quant_description.keys() and \
            self.quant_description['fa_quant_type'] is not None:
            return AscendKVCacheMethod(self, prefix)
        elif isinstance(layer, FusedMoE):
            if self.is_layer_skipped_ascend(prefix,
                                            self.packed_modules_mapping):
                return AscendUnquantizedFusedMoEMethod(layer.moe_config)
            return AscendFusedMoEMethod(self, prefix,
                                        self.packed_modules_mapping, layer)
        elif isinstance(layer, VocabParallelEmbedding):
            if self.is_layer_skipped_ascend(prefix,
                                            self.packed_modules_mapping):
                return UnquantizedEmbeddingMethod()
            return AscendEmbeddingMethod(self, prefix,
                                         self.packed_modules_mapping, layer)
        return None

    def is_layer_skipped_ascend(
        self,
        prefix: str,
        fused_mapping: Mapping[str, List[str]] = MappingProxyType({})):
        # adapted from vllm.model_executor.layers.quantization.utils.quant_utils.is_layer_skipped
        proj_name = prefix.split(".")[-1]
        if proj_name in fused_mapping:
            shard_prefixes = [
                prefix.replace(proj_name, shard_proj_name)
                for shard_proj_name in fused_mapping[proj_name]
            ]

            is_skipped = None
            for shard_prefix in shard_prefixes:
                is_shard_skipped = self.quant_description[shard_prefix +
                                                          '.weight'] == "FLOAT"

                if is_skipped is None:
                    is_skipped = is_shard_skipped
                elif is_shard_skipped != is_skipped:
                    raise ValueError(
                        f"Detected some but not all shards of {prefix} "
                        "are quantized. All shards of fused layers "
                        "to have the same precision.")
        else:
            is_skipped = self.quant_description[prefix + '.weight'] == "FLOAT"

        assert is_skipped is not None
        return is_skipped

    def get_scaled_act_names(self) -> List[str]:
        return []


# Remove the original compressed_tensors method to replace with our implementation
def _remove_quantization_method():
    if COMPRESSED_TENSORS_METHOD in QUANTIZATION_METHODS:
        QUANTIZATION_METHODS.remove(COMPRESSED_TENSORS_METHOD)


_remove_quantization_method()

QUANTIZATION_SCHEME_MAP_TYPE = dict[str, Optional[dict[str, "QuantizationArgs"]]]


@register_quantization_config(COMPRESSED_TENSORS_METHOD)
class AscendCompressedTensorsConfig(QuantizationConfig):
    """Config class for LLM-Compressor (compressed_tensors) quantization on Ascend.
    
    This class adapts the compressed_tensors format to work with Ascend's
    quantization implementations.
    """

    def __init__(
        self,
        target_scheme_map: dict[str, Any],
        ignore: list[str],
        quant_format: str,
        config: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.ignore = ignore
        self.quant_format = quant_format
        # Map from [target -> scheme]
        self.target_scheme_map = target_scheme_map
        self.quant_description = config

    def get_name(self) -> str:
        return "compressed-tensors"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.int8, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            "Ascend hardware dose not support \"get_min_capability\" feature.")

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str,
                                      Any]) -> "AscendCompressedTensorsConfig":
        ignore: list[str] = cast(list[str], config.get("ignore", []))
        quant_format = cast(str, config.get("format"))
        target_scheme_map = cls._quantization_scheme_map_from_config(
            config=config)

        return cls(
            target_scheme_map=target_scheme_map,
            ignore=ignore,
            quant_format=quant_format,
            config=config,
        )

    @classmethod
    def _quantization_scheme_map_from_config(
            cls, config: dict[str, Any]) -> QUANTIZATION_SCHEME_MAP_TYPE:
        """Build target scheme map from config.
        
        :param config: The `quantization_config` dictionary from config.json
        :return: A dictionary mapping target layer names to their corresponding
            quantization_args for weights and input activations
        """

        target_scheme_map: dict[str, Any] = dict()
        quant_format = cast(str, config.get("format"))

        config_groups = config.get("config_groups", dict())
        for _, quant_config in config_groups.items():
            targets = quant_config.get("targets")
            for target in targets:
                target_scheme_map[target] = {}
                target_scheme_map[target][
                    "weights"] = QuantizationArgs.model_validate(
                        quant_config.get("weights"))

                target_scheme_map[target]["input_activations"] = None
                target_scheme_map[target]["format"] = quant_config.get(
                    "format")
                format = target_scheme_map[target].get("format")
                # If no per-config format defined, use global format in config
                act_quant_format = (
                    is_activation_quantization_format(format)
                    if format is not None else
                    is_activation_quantization_format(quant_format))
                input_activations = quant_config.get("input_activations")
                if act_quant_format and input_activations is not None:
                    target_scheme_map[target]["input_activations"] = (
                        QuantizationArgs.model_validate(
                            quant_config.get("input_activations")))
        return target_scheme_map

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            layer.ascend_quant_method = COMPRESSED_TENSORS_METHOD
            # collect schemes
            quant_scheme = self.get_scheme(layer=layer, layer_name=prefix)

            # choose quantization method
            quant_method = UnquantizedLinearMethod()
            if quant_scheme is not None:
                layer.scheme = quant_scheme
                ascend_quant_config = AscendQuantConfig(self.quant_description
                                                        or {})
                quant_method = AscendLinearMethod(ascend_quant_config, prefix,
                                                  None, layer)
            return quant_method
        if isinstance(layer, FusedMoE):
            layer.ascend_quant_method = COMPRESSED_TENSORS_METHOD
            # collect schemes
            quant_scheme = self.get_scheme(layer=layer, layer_name=prefix)

            # choose quantization method
            quant_method = AscendUnquantizedFusedMoEMethod(layer.moe_config)
            if quant_scheme is not None:
                layer.scheme = quant_scheme
                ascend_quant_config = AscendQuantConfig(self.quant_description
                                                        or {})
                quant_method = AscendFusedMoEMethod(
                    ascend_quant_config, prefix,
                    ascend_quant_config.packed_modules_mapping, layer)
            return quant_method
        return None

    def get_scheme(self,
                   layer: torch.nn.Module,
                   layer_name: Optional[str] = None
                   ) -> Optional["CompressedTensorsScheme"]:
        """Get the quantization scheme for a layer.
        
        compressed-tensors supports non uniform in the following way:

        targets of config_groups: There can be N config_groups which each
            have a quantization scheme. Each config_group has a list of targets
            which can be a full layer_name, a regex for a layer_name, or
            an nn.Module name.

        Detect whether a layer_name is found in any target and
        use the quantization scheme corresponding to the matched target
        to select the CompressedTensorsScheme used for inference.
        """

        # Find the "target" in the compressed-tensors config
        # that our layer conforms to.
        if should_ignore_layer(layer_name,
                               ignore=self.ignore,
                               fused_mapping=self.packed_modules_mapping):
            return None

        # Will be empty for models with only sparsity
        weight_quant = input_quant = None
        if self.target_scheme_map:
            matched_target = find_matched_target(
                layer_name=layer_name,
                module=layer,
                targets=self.target_scheme_map.keys(),
                fused_mapping=self.packed_modules_mapping,
            )

            scheme_dict = self.target_scheme_map[matched_target]
            weight_quant = scheme_dict.get("weights")
            input_quant = scheme_dict.get("input_activations")

        if weight_quant is None:
            logger.warning_once("Acceleration for non-quantized schemes is "
                                "not supported by Compressed Tensors. "
                                "Falling back to UnquantizedLinearMethod")
            return None

        else:
            # Find the quant_scheme
            scheme = self._get_scheme_from_parts(
                weight_quant=weight_quant,
                input_quant=input_quant,
            )
        return scheme

    def _get_scheme_from_parts(
            self, weight_quant: "QuantizationArgs",
            input_quant: "QuantizationArgs") -> "CompressedTensorsScheme":
        """Determine the appropriate scheme based on quantization args."""
        from .methods import (
            AscendW8A8LinearMethod,
            AscendW8A8DynamicLinearMethod,
            AscendW4A16FusedMoEMethod,
        )
        
        act_quant_format = is_activation_quantization_format(self.quant_format)
        if act_quant_format and input_quant is not None:
            if self._is_static_tensor_w8a8(weight_quant, input_quant):
                return AscendW8A8LinearMethod()

            if self._is_dynamic_token_w8a8(weight_quant, input_quant):
                return AscendW8A8DynamicLinearMethod()

        if weight_quant is not None:
            if self._is_w4a16(weight_quant):
                return AscendW4A16FusedMoEMethod()

        raise NotImplementedError(
            "No compressed-tensors compatible scheme was found.")

    def _is_static_tensor_w8a8(self, weight_quant: "QuantizationArgs",
                               input_quant: "QuantizationArgs") -> bool:
        is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.CHANNEL.value)
        is_tensor = (weight_strategy and input_quant.strategy
                     == QuantizationStrategy.TENSOR.value)
        is_static = not weight_quant.dynamic and not input_quant.dynamic
        is_symmetric = weight_quant.symmetric and input_quant.symmetric

        # Only symmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_8_bits and is_tensor and is_symmetric and is_static

    def _is_dynamic_token_w8a8(self, weight_quant: "QuantizationArgs",
                               input_quant: "QuantizationArgs") -> bool:
        is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.CHANNEL.value)
        is_token = (weight_strategy and input_quant.strategy
                    == QuantizationStrategy.TOKEN.value)
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic
        is_symmetric = weight_quant.symmetric and input_quant.symmetric

        # Only symmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_8_bits and is_token and is_symmetric and is_dynamic

    def _is_w4a16(self, weight_quant: "QuantizationArgs") -> bool:
        is_4_bits = weight_quant.num_bits == 4
        return is_4_bits

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        self.target_scheme_map = hf_to_vllm_mapper.apply_dict(
            self.target_scheme_map)
        self.ignore = hf_to_vllm_mapper.apply_list(self.ignore)
