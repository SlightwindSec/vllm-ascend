# MTP 第二次主模型 Forward 调试日志说明

## 背景与目的

在图模式（ACL Graph）+ MTP 投机解码场景下，**第二次主模型 forward**（即主模型第一次做 decode）有时会出现：

1. **非确定性**：同一输入多次运行，`logits.argmax` 结果不一致，本应确定的 greedy 采样出现随机性。
2. **精度异常**：第一个有效 token 的预测不合理（如 "I" 后跟 ","），导致乱码。

为对比 **eager 模式** 与 **图模式** 下第二次主模型推理的差异，需要详细记录该步的输入、attention 元数据与输出，并可选地将关键 tensor 保存到本地，便于用 `torch.load()` 做离线对比（例如 kvcache、pad 区域对精度的影响）。

本文档说明新增的调试日志与保存逻辑、使用方式，以及各条日志的含义与排查建议。

## 启用方式

- **调试日志**（与原有 MTP 调试共用开关）  
  ```bash
  VLLM_MTP_DEBUG=1 VLLM_MTP_DEBUG_DIR=/path/to/dir python tests/qwen3_next_mtp.py
  ```
  会多写一份**第二次主模型 forward 专用**日志文件（见下文）。

- **保存 tensor 到本地**（可选）  
  ```bash
  VLLM_MTP_DEBUG=1 VLLM_MTP_DEBUG_DIR=/path/to/dir VLLM_MTP_DEBUG_SAVE_TENSORS=1 python tests/qwen3_next_mtp.py
  ```
  在第二次主模型 forward 时，将输入/输出 tensor 以 `torch.save` 形式写入 `VLLM_MTP_DEBUG_DIR`，便于与 eager/图模式分别跑一次后对比。

## 日志文件与保存文件

### 1. 原有 MTP 调试日志（不变）

- 路径：`{VLLM_MTP_DEBUG_DIR}/mtp_debug_tp{tp}_dp{dp}_pp{pp}_rank{rank}.log`
- 内容：`prepare_inputs`、`attn_metadata_before_forward`、`pre_compute_logits`、`post_compute_logits`、`sample_output` 等（与之前一致）。

### 2. 第二次主模型 forward 的 tag（复用同一日志文件）

- 与原有 MTP 调试共用同一文件：`{VLLM_MTP_DEBUG_DIR}/mtp_debug_tp{tp}_dp{dp}_pp{pp}_rank{rank}.log`，文件名保持一致。
- 触发条件：当前步为**主模型第二次 forward**（即第一次 decode），且 `VLLM_MTP_DEBUG=1`。  
  判定方式：`use_spec_decode and num_computed_tokens_cpu[0] > 0 and _mtp_forward_step_counter == 2`。
- 建议用法：  
  - 用**相同输入**分别跑 **eager**（`enforce_eager=True`）和**图模式**（如 `cudagraph_capture_sizes=[4]`），各生成一份 `mtp_debug_*.log`，按 tag 过滤出 `second_forward_*` 相关行逐条对比差异。

### 3. Tensor 保存文件（VLLM_MTP_DEBUG_SAVE_TENSORS=1 时）

所有文件名均带 `second_forward_{eager|graph}_step2_`，便于区分模式与步数。

| 文件名 | 内容 | 用途 |
|--------|------|------|
| `second_forward_{eager\|graph}_step2_input_ids.pt` | 本步输入 token ids（含 pad 到 `num_tokens_padded`） | 确认 eager/图 输入是否一致 |
| `second_forward_{eager\|graph}_step2_positions.pt` | 本步 position ids（同上长度） | 确认 position 是否一致，pad 位置是否异常 |
| `second_forward_{eager\|graph}_step2_slot_mapping.pt` | attn 元数据中的 slot_mapping（前 `num_tokens_padded`） | 与 KV cache 布局相关，影响精度 |
| `second_forward_{eager\|graph}_step2_block_tables.pt` | attn 元数据中的 block_tables | 同上 |
| `second_forward_{eager\|graph}_step2_hidden_states.pt` | forward 输出 hidden_states（前 `num_tokens_padded`） | 对比 eager/图 输出是否一致 |
| `second_forward_{eager\|graph}_step2_logits.pt` | `compute_logits(sample_hidden_states)` 的结果 | 对比 logits/argmax 差异 |

对比示例（本地脚本）：

```python
eager_hs = torch.load("second_forward_eager_step2_hidden_states.pt")
graph_hs = torch.load("second_forward_graph_step2_hidden_states.pt")
print("hidden_states diff:", (eager_hs - graph_hs).abs().max().item())
```

## 各 tag 含义与排查建议

以下 tag 均出现在同一 MTP 调试日志文件 `mtp_debug_*.log` 中（与 prepare_inputs、pre_compute_logits 等共用）。

### second_forward_step

- **含义**：当前是否为第二次主模型 forward、图模式、pad 信息。
- **字段**：`step`、`cudagraph_mode`、`num_tokens_unpadded`、`num_tokens_padded`、`num_reqs`、`pad_attn`、`num_pad_tokens`。
- **排查**：确认 `num_pad_tokens = num_tokens_padded - num_tokens_unpadded` 在图模式下是否非 0；`pad_attn` 在图 FULL 时为 True，与 pad 是否参与 attention 相关。

### second_forward_input_ids

- **含义**：本步输入 token ids，区分「有效长度」与「pad 区域」。
- **字段**：`input_ids_full`、`input_ids_unpadded`、`input_ids_padded_region`。
- **排查**：eager 与图模式应完全一致；若不一致，说明 pad 前逻辑或 scheduler 输出有差异。  
  **Pad 对精度的影响**：若 pad 区域 token id 或 position 与 eager 不一致，会改变 attention 的 Q/K/V，进而影响有效位置的 hidden 与 logits。

### second_forward_positions

- **含义**：本步 position ids，同样区分有效与 pad 区域。
- **字段**：`positions_full`、`positions_unpadded`、`positions_padded_region`。
- **排查**：pad 区域在图模式下常为「续写」的 position（例如 3, 4...），需确认与 eager 是否一致；position 错误会直接导致 RoPE 和 attention 错位。

### second_forward_logits_indices

- **含义**：`compute_logits` 时从 hidden_states 中取哪些位置（对应哪些 token）。
- **字段**：`logits_indices`。
- **排查**：应与 `num_tokens_unpadded`、spec decode 的 logits 选取逻辑一致；若与 eager 不一致，会导致采样位置错位。

### second_forward_attn_metadata

- **含义**：当前步 attention 元数据，与「pad 如何参与 attention」强相关。
- **字段**：  
  - `seq_lens_list`：各请求当前长度（含本步）。  
  - `actual_seq_lengths_q`：MTP 下用于 attention 的「实际 query 长度」列表，**若与 eager 不一致极易导致精度问题**。  
  - `slot_mapping_full` / `slot_mapping_padded_region`：KV cache 槽位映射，pad 位置若为 -1 表示不写 cache。  
  - `block_tables_first_row`、`query_start_loc`、`num_actual_tokens` 等。
- **排查**：  
  - 重点对比 **actual_seq_lengths_q**：图模式若因 pad 或 capture 导致与 eager 不同，会改变 attention 的有效长度与 mask。  
  - **slot_mapping_padded_region**：若 pad 位置未置 -1 或与 eager 不一致，可能让 pad 位置参与 KV 读写，污染结果。

### second_forward_num_computed_tokens

- **含义**：各请求已计算的 token 数（到本步之前）。
- **字段**：`num_computed_tokens_req0`、`num_computed_tokens_all`。
- **排查**：用于确认「第二次 forward」的语义（已算过 prefill + 1 个 decode token 等）。

### second_forward_batch_slot_mapping

- **含义**：`input_batch.block_table` 中第一个 block_table 的 slot_mapping（前 `num_tokens_padded`）。
- **排查**：与 attn 的 slot_mapping 对照，确认 batch 与 attn 元数据是否一致。

### second_forward_model_kwargs_keys

- **含义**：传入 `model(**model_kwargs)` 的 key 列表（不含 value）。
- **排查**：确认 eager/图 传入的 key 一致（例如是否都含 kv_cache 等）；若需进一步对比 kvcache 数值，可结合 `VLLM_MTP_DEBUG_SAVE_TENSORS` 扩展保存。

### second_forward_hidden_states

- **含义**：本步 forward 输出的 hidden 统计（仅第二次 forward 时打）。
- **字段**：`hidden_states_shape`、`hs_norms_first_4`、`unpadded_mean_norm`、`padded_mean_norm`。
- **排查**：  
  - 对比 eager/图 的 `hs_norms_first_4`、`unpadded_mean_norm`、`padded_mean_norm`。  
  - **Pad 对精度的影响**：若图模式在 pad 位置的 hidden 与 eager 差异大，或 `padded_mean_norm` 异常，说明 pad 路径上的计算与 eager 不一致（例如 attention 对 pad 的处理不同）。

### second_forward_logits_argmax

- **含义**：本步 `logits.argmax(dim=-1)` 的完整结果与前 4 个。
- **字段**：`argmax_all`、`argmax_first_4`。
- **排查**：  
  - 同一输入下，eager 与图模式应得到相同 argmax；若不同，说明图模式在该步存在非确定性或精度问题。  
  - 结合 `second_forward_hidden_states` 和保存的 `logits.pt`，可定位差异来自 hidden 还是 lm_head。

### second_forward_save_error / second_forward_save_output_error

- **含义**：保存 tensor 时的异常信息。
- **排查**：若出现，检查磁盘空间与 `VLLM_MTP_DEBUG_DIR` 写权限。

## Pad 如何影响精度（小结）

- **输入侧**：pad 的 `input_ids` / `positions` 若与 eager 不一致，会改变 embedding 与 RoPE，进而改变 Q/K/V。  
- **Attention 元数据**：`actual_seq_lengths_q`、`slot_mapping`（含 pad 区域）若在图模式下与 eager 不一致，会改变有效长度、mask 和 KV 读写位置，导致同一 token 看到不同的 context。  
- **输出侧**：pad 位置的 hidden 在图模式下若计算方式不同（例如被 mask 方式不同），可能通过后续层或共享 buffer 间接影响有效位置的 logits。  

因此日志中刻意区分了「unpadded / padded 区域」的 input_ids、positions、slot_mapping、hidden norms，并建议同时对比 eager 与图模式的 `actual_seq_lengths_q` 与保存的 tensor，以精确定位 pad 在图模式下的差异点。

## 代码位置

- 写日志与保存：`vllm_ascend/worker/model_runner_v1.py`  
  - 复用 `_init_mtp_debug_log`、`_log_mtp_debug`，文件名与原有 MTP 调试一致（`mtp_debug_*.log`）。  
  - `execute_model` 内：`is_second_main_forward` 判定、第二次 forward 前后的 `_log_mtp_debug` 调用与 `torch.save`。

## 与 _log_mtp_debug 的关系

- **复用**：第二次主模型 forward 的详细日志通过 `_log_mtp_debug` 写入同一文件 `mtp_debug_*.log`，tag 以 `second_forward_*` 开头，便于 grep 过滤。  
- **开关**：与原有 MTP 调试共用 `VLLM_MTP_DEBUG=1`；tensor 保存另受 `VLLM_MTP_DEBUG_SAVE_TENSORS=1` 控制。
