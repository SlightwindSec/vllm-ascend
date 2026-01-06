#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

from typing import Any, Dict, Optional

import torch
from vllm.logger import logger

from vllm_ascend.utils import COMPRESSED_TENSORS_METHOD

# Import the registry and method map builder
from .methods import build_quant_method_map, get_scheme_class


def get_linear_quant_type(quant_description: Dict[str, Any], prefix: str,
                          packed_modules_mapping: Dict[str, Any]) -> str:
    """Determine the quantization type for a linear layer.
    
    Args:
        quant_description: The quantization description dictionary.
        prefix: The layer prefix.
        packed_modules_mapping: Mapping for packed/fused modules.
        
    Returns:
        The quantization type string (e.g., "W8A8_DYNAMIC").
    """
    proj_name = prefix.split(".")[-1]
    if proj_name in packed_modules_mapping:
        quant_type = None
        shard_prefixes = [
            prefix.replace(proj_name, shard_proj_name)
            for shard_proj_name in packed_modules_mapping[proj_name]
        ]
        for shard_prefix in shard_prefixes:
            shard_quant_type = quant_description[shard_prefix + '.weight']

            if quant_type is None:
                quant_type = shard_quant_type
            elif shard_quant_type != quant_type:
                raise ValueError(
                    f"Not all shards of {prefix} are quantized with same quant type."
                    f"Shard {proj_name} uses {shard_quant_type}, but another shard"
                    f"use {quant_type}. Please check quantization config.")
    else:
        quant_type = quant_description[prefix + '.weight']
    return quant_type


def get_quant_method(quant_description: Dict[str, Any],
                     prefix: str,
                     layer_type: str,
                     packed_modules_mapping: Optional[Dict[str, Any]] = None,
                     layer: Optional[torch.nn.Module] = None):
    """Get the appropriate quantization method for a layer.
    
    Args:
        quant_description: The quantization description dictionary.
        prefix: The layer prefix.
        layer_type: The type of layer ("linear", "moe", "attention").
        packed_modules_mapping: Mapping for packed/fused modules.
        layer: The layer module (optional).
        
    Returns:
        An instance of the appropriate quantization method class.
    """
    if quant_description.get("quant_method") == COMPRESSED_TENSORS_METHOD:
        return get_quant_method_llmcompressor(layer)

    return get_quant_method_modelslim(quant_description, prefix, layer_type,
                                      packed_modules_mapping)


def get_quant_method_llmcompressor(layer: torch.nn.Module):
    """Get quantization method for LLM-Compressor models.
    
    Args:
        layer: The layer module with a scheme attribute.
        
    Returns:
        The scheme from the layer.
    """
    logger.info_once("Using the vLLM Ascend llmcompressor Quantization now!")
    if layer.scheme is None:
        raise ValueError("A scheme must be defined for each layer")
    return layer.scheme


def get_quant_method_modelslim(
        quant_description: Dict[str, Any],
        prefix: str,
        layer_type: str,
        packed_modules_mapping: Optional[Dict[str, Any]] = None):
    """Get quantization method for ModelSlim models.
    
    Args:
        quant_description: The quantization description dictionary.
        prefix: The layer prefix.
        layer_type: The type of layer ("linear", "moe", "attention").
        packed_modules_mapping: Mapping for packed/fused modules.
        
    Returns:
        An instance of the appropriate quantization method class.
    """
    logger.info_once("Using the vLLM Ascend modelslim Quantization now!")
    if packed_modules_mapping is None:
        packed_modules_mapping = dict()
    # Attention
    if '.attn' in prefix and 'fa_quant_type' in quant_description.keys():
        quant_type = quant_description['fa_quant_type']
    # Linear
    else:
        quant_type = get_linear_quant_type(quant_description, prefix,
                                           packed_modules_mapping)

    # Use registry to get scheme class
    method_cls = get_scheme_class(quant_type, layer_type)
    if method_cls is not None:
        return method_cls()

    # Fall back to method map for backward compatibility
    method_map = build_quant_method_map()
    if quant_type in method_map:
        if layer_type in method_map[quant_type]:
            return method_map[quant_type][layer_type]()
        else:
            raise NotImplementedError(
                f"Currently, vLLM Ascend doesn't support {quant_type} for {layer_type}."
            )
    raise NotImplementedError(
        f"Currently, vLLM Ascend only supports following quant types: "
        f"{list(method_map.keys())}")


# For backward compatibility, also export the method map
def get_ascend_quantization_method_map() -> Dict[str, Dict[str, type]]:
    """Get the quantization method map for backward compatibility.
    
    Returns:
        Dictionary mapping quant_type -> {layer_type -> SchemeClass}.
    """
    return build_quant_method_map()


# Alias for backward compatibility
ASCEND_QUANTIZATION_METHOD_MAP = property(
    lambda self: build_quant_method_map())
