import math
from contextlib import contextmanager
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_dp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import (BatchDescriptor, get_forward_context,
                                  set_forward_context)

import vllm_ascend.envs as envs_ascend
from vllm_ascend.utils import (enable_sp, flashcomm2_enable, has_layer_idx,
                               is_moe_model)

if TYPE_CHECKING:
    from vllm_ascend.ops.weight_prefetch import WeightPrefetchMethod
else:
    WeightPrefetchMethod = None


class MoECommType(Enum):
    ALLGATHER = 0
    MC2 = 1
    ALLTOALL = 2
    FUSED_ALLTOALL = 3


# Threshold for enabling sequence parallelism (empirical value)
_SP_THRESHOLD_TOKENS = 1000
# Threshold for enabling MLP weight prefetch
_PREFETCH_TOKEN_THRESHOLD = 500


@contextmanager
def set_ascend_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
    num_tokens: Optional[int] = None,
    num_tokens_across_dp: Optional[torch.Tensor] = None,
    with_prefill: bool = True,
    in_profile_run: bool = False,
    reserved_mc2_mask: Optional[torch.Tensor] = None,
    moe_comm_type: Optional[MoECommType] = None,
    num_actual_tokens: Optional[int] = None,
    aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: Optional[BatchDescriptor] = None,
    prefetch_stream: Optional[torch.npu.Stream] = None,
    model_instance: Optional[torch.nn.Module] = None,
    weight_prefetch_method: Optional["WeightPrefetchMethod"] = None,
    is_mtp_model: bool = False,
):
    """Context manager extending vLLM's set_forward_context with Ascend-specific settings."""
    with set_forward_context(
            attn_metadata,
            vllm_config,
            virtual_engine=virtual_engine,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=aclgraph_runtime_mode,
            batch_descriptor=batch_descriptor,
    ):
        forward_context = get_forward_context()
        tp_world_size = get_tensor_model_parallel_world_size()

        # MoE communication setup
        from vllm_ascend.ops.fused_moe.moe_comm_method import \
            get_moe_comm_method
        forward_context.moe_comm_type = moe_comm_type
        forward_context.moe_comm_method = get_moe_comm_method(moe_comm_type)
        forward_context.with_prefill = with_prefill
        forward_context.in_profile_run = in_profile_run
        forward_context.capturing = False
        forward_context.is_mtp_model = is_mtp_model

        # Sequence parallelism setup
        if is_moe_model(vllm_config):
            sp_enabled = enable_sp(vllm_config) and num_tokens is not None
            mmrs_fusion = False
        else:
            sp_enabled = (enable_sp(vllm_config) and num_tokens is not None
                          and num_tokens > _SP_THRESHOLD_TOKENS)
            mmrs_fusion = True

        flashcomm_v2_enabled = (flashcomm2_enable() and tp_world_size > 1
                                and num_tokens is not None)

        forward_context.sp_enabled = sp_enabled
        forward_context.flashcomm_v2_enabled = flashcomm_v2_enabled
        forward_context.mmrs_fusion = mmrs_fusion
        forward_context.num_tokens = num_tokens

        if sp_enabled or flashcomm_v2_enabled:
            forward_context.pad_size = (
                tp_world_size - (num_tokens % tp_world_size)) % tp_world_size

        # Layer tracking for optimization features
        forward_context.is_first_layer = True
        forward_context.layer_idx = (model_instance.model.start_layer if
                                     has_layer_idx(model_instance) else None)

        # Weight prefetch setup
        prefetch_mlp_enabled = (envs_ascend.VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE
                                and envs_ascend.VLLM_ASCEND_ENABLE_PREFETCH_MLP
                                and forward_context.layer_idx is not None
                                and num_tokens is not None
                                and num_tokens < _PREFETCH_TOKEN_THRESHOLD)
        forward_context.prefetch_mlp_enabled = prefetch_mlp_enabled
        forward_context.model_instance = model_instance
        forward_context.weight_prefetch_method = weight_prefetch_method
        if prefetch_mlp_enabled:
            forward_context.prefetch_stream = prefetch_stream
            forward_context.prefetch_mlp_gate_up_proj = False
            forward_context.prefetch_mlp_down_proj = False

        # Resolve num_tokens from attention metadata if needed
        resolved_num_tokens = num_tokens
        if resolved_num_tokens is None and attn_metadata is not None:
            resolved_num_tokens = getattr(attn_metadata, 'num_actual_tokens',
                                          None)

        # Data parallel padding calculations
        dp_world_size = get_dp_group().world_size
        if dp_world_size > 1 and forward_context.dp_metadata is not None:
            max_tokens_across_dp = forward_context.dp_metadata.max_tokens_across_dp_cpu.item(
            )
            if sp_enabled or flashcomm_v2_enabled:
                padded_length = (max_tokens_across_dp + tp_world_size -
                                 1) // tp_world_size * tp_world_size
                forward_context.padded_length = padded_length
                forward_context.pad_size = padded_length - (resolved_num_tokens
                                                            or 0)
        else:
            max_tokens_across_dp = resolved_num_tokens

        forward_context.max_tokens_across_dp = max_tokens_across_dp

        # MC2 padding and mask
        if resolved_num_tokens is not None:
            actual_tokens = num_actual_tokens if num_actual_tokens is not None else resolved_num_tokens
            padded_num_tokens = math.ceil(
                (max_tokens_across_dp or 0) / tp_world_size) * tp_world_size
            forward_context.padded_num_tokens = padded_num_tokens

            if reserved_mc2_mask is not None:
                mc2_mask = reserved_mc2_mask[:padded_num_tokens]
                mc2_mask[:actual_tokens] = True
                mc2_mask[actual_tokens:] = False
                forward_context.mc2_mask = mc2_mask

        try:
            yield
        finally:
            pass
