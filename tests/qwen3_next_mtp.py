import contextlib
import gc
import os
import sys

import torch
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig
from vllm.inputs import TextPrompt

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL_PATH = "/home/weight/Qwen3-Next-80B-A3B-Instruct"


def _cleanup_dist_env_and_memory():
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
    with contextlib.suppress(Exception):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.npu.empty_cache()

def _run_mtp_test(model_name: str):
    capture_sizes = [4]
    example_prompts = [
        "Hello,",
    ]

    llm = LLM(
        model=model_name,
        tensor_parallel_size=4,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        distributed_executor_backend="mp",
        # enforce_eager=True,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 1,
            "disable_padded_drafter_batch": False,
        },
        compilation_config=CompilationConfig(
            cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=capture_sizes,
        ),
    )

    inputs = [TextPrompt(prompt=p) for p in example_prompts]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=20)
    req_outputs = llm.generate(inputs, sampling_params=sampling_params)

    spec_outputs = []
    for req_output in req_outputs:
        sample = req_output.outputs[0]
        output_ids = list(req_output.prompt_token_ids) + list(sample.token_ids)
        output_str = (req_output.prompt or "") + sample.text
        spec_outputs.append((output_ids, output_str))

    for i, (_, text) in enumerate(spec_outputs):
        print(f"[{i}] {text!r}")

    del llm
    try:
        from vllm_ascend.ascend_config import clear_ascend_config
        clear_ascend_config()
    except ImportError:
        pass
    _cleanup_dist_env_and_memory()
    return spec_outputs

if __name__ == "__main__":
    _run_mtp_test(MODEL_PATH)
