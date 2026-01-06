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
"""Backward compatibility module.

This module re-exports classes from their new locations for backward compatibility.
New code should import from vllm_ascend.quantization.config and 
vllm_ascend.quantization.wrappers instead.
"""

# Re-export config classes
from vllm_ascend.quantization.config import AscendQuantConfig

# Re-export wrapper classes
from vllm_ascend.quantization.wrappers import (
    AscendLinearMethod,
    AscendKVCacheMethod,
    AscendFusedMoEMethod,
    AscendEmbeddingMethod,
)

# Re-export model mappings
from vllm_ascend.quantization.model_mappings import (
    packed_modules_model_mapping,
    QUANT_MODEL_PREFIX_MAPPINGS,
)

__all__ = [
    "AscendQuantConfig",
    "AscendLinearMethod",
    "AscendKVCacheMethod",
    "AscendFusedMoEMethod",
    "AscendEmbeddingMethod",
    "packed_modules_model_mapping",
    "QUANT_MODEL_PREFIX_MAPPINGS",
]
