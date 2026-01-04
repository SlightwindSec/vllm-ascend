from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch_npu

from vllm_ascend.attention.mla_v1 import (
    AscendMLAMetadata,
    DecodeMLAPreprocessResult,
    PrefillMLAPreprocessResult,
)
from vllm_ascend.attention.utils import trans_rope_weight, transdata
from vllm_ascend.quantization.w8a8 import AscendW8A8LinearMethod


def process_weights_for_mlapo(
    fused_qkv_a_proj,
    q_proj,
    q_a_layernorm,
    kv_a_layernorm,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_heads: int,
    qk_nope_head_dim: int,
    act_dtype: torch.dtype,
    use_mla_beta1_logic: bool = False,
) -> dict:
    kv_a_proj_wt = fused_qkv_a_proj.weight.data[
        ..., q_lora_rank:].contiguous()
    q_a_proj_wt = fused_qkv_a_proj.weight.data[
        ..., :q_lora_rank].contiguous()

    kv_a_proj_wt = kv_a_proj_wt.t().contiguous()
    kv_a_proj_wt = trans_rope_weight(kv_a_proj_wt, qk_rope_head_dim)
    kv_a_proj_wt = kv_a_proj_wt.t().contiguous()

    wd_qkv = torch.cat((kv_a_proj_wt, q_a_proj_wt), dim=-1)
    wd_qkv = wd_qkv.t().contiguous()
    wd_qkv = transdata(wd_qkv, block_size=(16, 32)).unsqueeze(0).contiguous()
    wd_qkv = torch_npu.npu_format_cast(wd_qkv, 29)

    kv_a_proj_deq_scl = fused_qkv_a_proj.deq_scale[
        q_lora_rank:].contiguous()
    q_a_proj_deq_scl = fused_qkv_a_proj.deq_scale[:q_lora_rank].contiguous()

    kv_a_proj_deq_scl = kv_a_proj_deq_scl.reshape(
        kv_lora_rank + qk_rope_head_dim, -1).contiguous()
    kv_a_proj_deq_scl = trans_rope_weight(kv_a_proj_deq_scl, qk_rope_head_dim)
    kv_a_proj_deq_scl = kv_a_proj_deq_scl.view(
        kv_lora_rank + qk_rope_head_dim).contiguous()
    deq_scale_qkv = torch.cat((kv_a_proj_deq_scl, q_a_proj_deq_scl),
                              dim=-1).contiguous()

    kv_a_proj_qt_bias = fused_qkv_a_proj.quant_bias[
        q_lora_rank:].contiguous()
    q_a_proj_qt_bias = fused_qkv_a_proj.quant_bias[:q_lora_rank].contiguous()

    kv_a_proj_qt_bias = kv_a_proj_qt_bias.reshape(
        kv_lora_rank + qk_rope_head_dim, -1).contiguous()
    kv_a_proj_qt_bias = trans_rope_weight(kv_a_proj_qt_bias, qk_rope_head_dim)
    kv_a_proj_qt_bias = kv_a_proj_qt_bias.view(
        kv_lora_rank + qk_rope_head_dim).contiguous()
    quant_bias_qkv = torch.cat((kv_a_proj_qt_bias, q_a_proj_qt_bias),
                               dim=-1).contiguous()

    wu_q = q_proj.weight.data
    wu_q = wu_q.t().reshape(num_heads,
                           qk_nope_head_dim + qk_rope_head_dim,
                           -1)
    wu_q = trans_rope_weight(wu_q, qk_rope_head_dim)
    wu_q = wu_q.reshape(
        num_heads * (qk_nope_head_dim + qk_rope_head_dim),
        -1)
    wu_q = transdata(wu_q, block_size=(16, 32)).unsqueeze(0).contiguous()
    wu_q = torch_npu.npu_format_cast(wu_q, 29)

    qb_deq_scl = q_proj.deq_scale.data
    qb_deq_scl = qb_deq_scl.reshape(
        num_heads, qk_nope_head_dim + qk_rope_head_dim, -1)
    qb_deq_scl = trans_rope_weight(qb_deq_scl, qk_rope_head_dim)
    qb_deq_scl = qb_deq_scl.reshape(
        num_heads * (qk_nope_head_dim + qk_rope_head_dim))

    qb_qt_bias = q_proj.quant_bias.data
    qb_qt_bias = qb_qt_bias.reshape(
        num_heads, qk_nope_head_dim + qk_rope_head_dim, -1)
    qb_qt_bias = trans_rope_weight(qb_qt_bias, qk_rope_head_dim)
    qb_qt_bias = qb_qt_bias.reshape(
        num_heads * (qk_nope_head_dim + qk_rope_head_dim))

    device = q_proj.weight.device
    gamma1 = q_a_layernorm.weight.data
    if use_mla_beta1_logic:
        # MLA logic: handle None bias
        beta1 = torch.zeros_like(gamma1) if (
            _bias := q_a_layernorm.bias) is None else _bias.data
    else:
        # SFA logic: direct access (assumes bias exists)
        beta1 = q_a_layernorm.bias.data
    gamma2 = kv_a_layernorm.weight.data

    quant_scale0 = fused_qkv_a_proj.input_scale.data
    quant_offset0 = fused_qkv_a_proj.input_offset.data
    quant_scale1 = q_proj.input_scale.data
    quant_offset1 = q_proj.input_offset.data

    ctkv_scale = torch.tensor([1], dtype=act_dtype, device=device)
    q_nope_scale = torch.tensor([1], dtype=act_dtype, device=device)

    return {
        'wd_qkv': wd_qkv,
        'wu_q': wu_q,
        'deq_scale_qkv': deq_scale_qkv,
        'quant_bias_qkv': quant_bias_qkv,
        'qb_deq_scl': qb_deq_scl,
        'qb_qt_bias': qb_qt_bias,
        'gamma1': gamma1,
        'beta1': beta1,
        'gamma2': gamma2,
        'quant_scale0': quant_scale0,
        'quant_offset0': quant_offset0,
        'quant_scale1': quant_scale1,
        'quant_offset1': quant_offset1,
        'ctkv_scale': ctkv_scale,
        'q_nope_scale': q_nope_scale,
    }


class MLAPOStrategy(ABC):

    @abstractmethod
    def can_enable(self, impl) -> Tuple[bool, list[str]]:
        pass

    @abstractmethod
    def process_weights(self, impl, act_dtype: torch.dtype):
        pass

    @abstractmethod
    def preprocess_decode(
        self,
        impl,
        hidden_states: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, ...],
        attn_metadata: AscendMLAMetadata,
    ) -> Tuple[Optional[DecodeMLAPreprocessResult],
               Optional[PrefillMLAPreprocessResult]]:
        pass


class MLAMLAPOStrategy(MLAPOStrategy):

    def can_enable(self, impl) -> Tuple[bool, list[str]]:
        """Check if MLAPO can be enabled for MLA."""
        reasons = []

        if impl.fused_qkv_a_proj is None or not isinstance(
            getattr(impl.fused_qkv_a_proj.quant_method, 'quant_method', None),
            AscendW8A8LinearMethod
        ):
            reasons.append(
                "Currently mlapo only supports W8A8 quantization in MLA scenario."
                "Some layers in your model are not quantized with W8A8,"
                "thus mlapo is disabled for these layers."
            )

        return len(reasons) == 0, reasons

    def process_weights(self, impl, act_dtype: torch.dtype):
        weights_dict = process_weights_for_mlapo(
            fused_qkv_a_proj=impl.fused_qkv_a_proj,
            q_proj=impl.q_proj,
            q_a_layernorm=impl.q_a_layernorm,
            kv_a_layernorm=impl.kv_a_layernorm,
            q_lora_rank=impl.q_lora_rank,
            kv_lora_rank=impl.kv_lora_rank,
            qk_rope_head_dim=impl.qk_rope_head_dim,
            num_heads=impl.num_heads,
            qk_nope_head_dim=impl.qk_nope_head_dim,
            act_dtype=act_dtype,
            use_mla_beta1_logic=True,  # MLA uses special beta1 logic
        )

        # Assign processed weights to implementation
        impl.wd_qkv = weights_dict['wd_qkv']
        impl.wu_q = weights_dict['wu_q']
        impl.deq_scale_qkv = weights_dict['deq_scale_qkv']
        impl.quant_bias_qkv = weights_dict['quant_bias_qkv']
        impl.qb_deq_scl = weights_dict['qb_deq_scl']
        impl.qb_qt_bias = weights_dict['qb_qt_bias']
        impl.gamma1 = weights_dict['gamma1']
        impl.beta1 = weights_dict['beta1']
        impl.gamma2 = weights_dict['gamma2']
        impl.quant_scale0 = weights_dict['quant_scale0']
        impl.quant_offset0 = weights_dict['quant_offset0']
        impl.quant_scale1 = weights_dict['quant_scale1']
        impl.quant_offset1 = weights_dict['quant_offset1']
        impl.ctkv_scale = weights_dict['ctkv_scale']
        impl.q_nope_scale = weights_dict['q_nope_scale']

    def preprocess_decode(
        self,
        impl,
        hidden_states: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, ...],
        attn_metadata: AscendMLAMetadata,
    ) -> Tuple[Optional[DecodeMLAPreprocessResult],
               Optional[PrefillMLAPreprocessResult]]:
        return impl._mla_preprocess_only_decode(hidden_states, kv_cache,
                                                attn_metadata)


class SFAMLAPOStrategy(MLAPOStrategy):
    
    def can_enable(self, impl) -> Tuple[bool, list[str]]:
        reasons = []

        quant_method = getattr(
            getattr(impl.fused_qkv_a_proj, "quant_method", None),
            "quant_method",
            None,
        )

        if impl.fused_qkv_a_proj is None or not isinstance(
            quant_method, AscendW8A8LinearMethod
        ):
            reasons.append(
                "Currently mlapo only supports W8A8 quantization in SFA scenario."
                "Some layers in your model are not quantized with W8A8,"
                "thus mlapo is disabled for these layers."
            )

        if impl.enable_sfa_cp:
            reasons.append(
                "Currently mlapo does not support SFA with CP,"
                "thus mlapo is disabled for these layers."
            )

        return len(reasons) == 0, reasons

    def process_weights(self, impl, act_dtype: torch.dtype):
        assert impl.kv_a_proj_with_mqa is None
        assert impl.fused_qkv_a_proj is not None

        weights_dict = process_weights_for_mlapo(
            fused_qkv_a_proj=impl.fused_qkv_a_proj,
            q_proj=impl.q_proj,
            q_a_layernorm=impl.q_a_layernorm,
            kv_a_layernorm=impl.kv_a_layernorm,
            q_lora_rank=impl.q_lora_rank,
            kv_lora_rank=impl.kv_lora_rank,
            qk_rope_head_dim=impl.qk_rope_head_dim,
            num_heads=impl.num_heads,
            qk_nope_head_dim=impl.qk_nope_head_dim,
            act_dtype=act_dtype,
            use_mla_beta1_logic=False,  # SFA uses direct access
        )

        impl.wd_qkv = weights_dict['wd_qkv']
        impl.wu_q = weights_dict['wu_q']
        impl.deq_scale_qkv = weights_dict['deq_scale_qkv']
        impl.quant_bias_qkv = weights_dict['quant_bias_qkv']
        impl.qb_deq_scl = weights_dict['qb_deq_scl']
        impl.qb_qt_bias = weights_dict['qb_qt_bias']
        impl.gamma1 = weights_dict['gamma1']
        impl.beta1 = weights_dict['beta1']
        impl.gamma2 = weights_dict['gamma2']
        impl.quant_scale0 = weights_dict['quant_scale0']
        impl.quant_offset0 = weights_dict['quant_offset0']
        impl.quant_scale1 = weights_dict['quant_scale1']
        impl.quant_offset1 = weights_dict['quant_offset1']
        impl.ctkv_scale = weights_dict['ctkv_scale']
        impl.q_nope_scale = weights_dict['q_nope_scale']

        if (impl.vllm_config.kv_transfer_config is not None and
            impl.vllm_config.kv_transfer_config.is_kv_consumer):
            impl.fused_qkv_a_proj.weight = None
            impl.fused_qkv_a_proj.deq_scale = None
            impl.fused_qkv_a_proj.quant_bias = None
            impl.q_proj.weight = None
            impl.q_proj.deq_scale = None
            impl.q_proj.quant_bias = None
            torch.npu.empty_cache()

    def preprocess_decode(
        self,
        impl,
        hidden_states: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, ...],
        attn_metadata: AscendMLAMetadata,
    ) -> Tuple[Optional[DecodeMLAPreprocessResult],
               Optional[PrefillMLAPreprocessResult]]:
        return None, None
