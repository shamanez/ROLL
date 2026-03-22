# Phase 5: Run Qwen3.5-2B SWE Pipeline - Solution

## Overview

This phase transitions from single-GPU-per-role (Phase 3/4) to full 8-GPU Mode B (partial GPU, simultaneous train+infer) using the Qwen3.5-2B model. The goal is to run the complete agentic SWE-bench pipeline with Megatron training on GPUs 0-3 (TP=2, CP=2) and vLLM inference across all 8 GPUs, with ROLL dynamically shrinking/expanding vLLM workers to share GPUs with Megatron.

---

## Problem 1: CUDA IPC Permission Error (`pidfd_getfd: Operation not permitted`)

### Symptom

When running Mode B (partial GPU mode), the pipeline crashed during weight transfer between Megatron train workers and vLLM infer workers on shared GPUs (0-3):

```
OSError: pidfd_getfd: Operation not permitted
```

### Root Cause

Mode B requires CUDA IPC (Inter-Process Communication) to transfer updated model weights from Megatron training processes to vLLM inference processes that share the same GPUs (0-3). CUDA IPC uses the `pidfd_getfd` system call to access file descriptors across processes. Docker's default seccomp profile blocks this syscall, and the default PID namespace isolation prevents cross-process FD access.

Three missing Docker flags were needed:
- **`--pid=host`**: Shares the host PID namespace so CUDA IPC can reference processes across containers
- **`--cap-add SYS_PTRACE`**: Allows `ptrace`-family syscalls including `pidfd_getfd`
- **`--security-opt seccomp=unconfined`**: Disables the seccomp filter that blocks `pidfd_getfd`

### Fix

Recreated the `roll_swe_runner` container with the additional flags:

```bash
# Stop and remove old container
docker stop roll_swe_runner && docker rm roll_swe_runner

# Recreate with CUDA IPC permissions
docker run -d \
  --name roll_swe_runner \
  --gpus all \
  --ipc=host \
  --pid=host \
  --cap-add CAP_SYS_ADMIN \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --security-opt label=disable \
  --network host \
  -v /home/ubuntu:/home/ubuntu \
  -v /var/run/docker.sock:/var/run/docker.sock \
  <image> \
  sleep infinity
```

**Key new flags compared to Phase 1 container:**
| Flag | Purpose |
|---|---|
| `--pid=host` | Share host PID namespace for CUDA IPC cross-process FD access |
| `--cap-add SYS_PTRACE` | Allow `pidfd_getfd` syscall |
| `--security-opt seccomp=unconfined` | Disable seccomp profile that blocks `pidfd_getfd` |

The existing flags (`--gpus all`, `--ipc=host`, `--cap-add CAP_SYS_ADMIN`, `--network host`, volume mounts) were retained.

---

## Problem 2: Container Rebuild Lost Installed Packages

### Symptom

After recreating the container, all previously installed packages (ROLL, mcore_adapter, rl-rock, terminal-bench-datasets) were missing because they were installed inside the old container's writable layer.

### Fix

Re-ran the full installation sequence inside the new container:

```bash
docker exec -it roll_swe_runner bash

# Inside container:
cd /home/ubuntu/ALE-latest/ROLL-personal

# Install ROLL and mcore_adapter
pip install -e .
pip install -e ./mcore_adapter

# Install ROCK SDK (for sandbox communication)
pip install rl-rock

# Install common requirements
pip install -r requirements_common.txt

# Re-clone terminal-bench-datasets (needed for SWE-bench test files)
cd /
git clone https://github.com/laude-institute/terminal-bench-datasets.git terminal-bench-datasets
```

This is the same sequence documented in `solutions/install_up_roll.sh`, just repeated for the new container.

---

## Problem 3: Batch Size / DP Validation Error (0.8B Config)

### Symptom

The 0.8B config (which uses TP=1 on 4 training GPUs) failed validation because `rollout_batch_size: 1` is not divisible by DP=4.

### Root Cause

With TP=1, PP=1, CP=1 on 4 GPUs, the data parallelism degree is:
```
DP = num_gpus / (TP * PP * CP) = 4 / (1 * 1 * 1) = 4
```

ROLL requires `rollout_batch_size % DP == 0`. A batch size of 1 is not divisible by 4.

Note: the 2B config does not have this problem because it uses TP=2, CP=2, giving DP = 4 / (2 * 1 * 2) = 1.

### Fix

Changed `rollout_batch_size` and `val_batch_size` from 1 to 4 in the 0.8B config:

**File:** `ROLL-personal/examples/agentic_demo/agent_val_rock_swe_qwen35_08b.yaml`
```yaml
# Before
rollout_batch_size: 1
val_batch_size: 1

# After
rollout_batch_size: 4
val_batch_size: 4
```

---

## The Jinja2 Fix (Verified)

### Background (Identified in Phase 4)

The Qwen3.5 chat template uses the Jinja2 `items()` filter to iterate over tool call arguments. When tool call arguments are JSON strings (e.g., `"{\"path\": \"/tmp/foo\"}"`) instead of Python dicts, the `items()` filter fails with a Jinja2 error because strings do not have an `items()` method.

### Root Cause

In the ROLL tokenization pipeline, tool call parameters were stored as raw JSON strings. The Qwen3.5 Jinja2 template expected them as dicts.

### Fix

Set `parse_tool_call_parameter_to_dict: true` in the config. This activates the parsing logic in `token_mask_utils.py` (lines 304-311), which calls `json.loads()` on tool call argument strings to convert them to Python dicts before passing them to the Jinja2 template.

**File:** `ROLL-personal/roll/utils/token_mask_utils.py` lines 304-311
```python
# When parse_tool_call_parameter_to_dict is true:
# Converts tool_call arguments from JSON strings to Python dicts
# so Jinja2 items() filter works correctly
```

### Verification

The 0.8B pipeline ran 6+ successful environment steps with ZERO Jinja2 template errors, confirming the fix works.

---

## Config Changes

### New File: `ROLL-personal/examples/agentic_demo/agent_val_rock_swe_qwen35_2b.yaml`

Full config for Qwen3.5-2B with Mode B (partial GPU, simultaneous train+infer).

**Key parameters:**

| Parameter | Value | Reason |
|---|---|---|
| `pretrain` | `/home/ubuntu/ALE-latest/model-checkpoints/Qwen3.5-2B` | 2B model checkpoint |
| `reward_pretrain` | `/home/ubuntu/ALE-latest/model-checkpoints/Qwen3.5-2B` | Match pretrain |
| `async_generation_ratio: 1` | Enables Mode B (partial GPU) | Simultaneous train+infer |
| `parse_tool_call_parameter_to_dict: true` | Jinja2 `items()` fix | Qwen3.5 template compatibility |
| `skip_mock_system_prompt: true` | Skip mock system prompt | Qwen3.5 template compatibility |
| `track_with: wandb` | WandB logging | Track training metrics |
| `rollout_batch_size: 1` | Batch size 1 | DP=1 (TP=2, CP=2 on 4 GPUs) |
| `val_batch_size: 1` | Val batch size 1 | DP=1 |

**GPU Allocation (Mode B - Partial):**

| Role | GPUs | Strategy | Parallelism | Notes |
|---|---|---|---|---|
| `actor_train` | 0-3 | `megatron_train` | TP=2, CP=2, SP=true, distributed_optimizer=true | Megatron training |
| `actor_infer` | 0-7 | `vllm` | TP=1, 8 DP workers, gpu_memory_utilization=0.6 | vLLM inference |
| `reference` | 0-3 | `megatron_infer` | TP=2, CP=2 | Reference model for KL |

**How Mode B Works:**

1. **Inference phase**: vLLM uses all 8 GPUs (8 DP workers, TP=1 each) for trajectory generation
2. **Training phase**: vLLM shrinks off GPUs 0-3 (offloads model/KV cache to CPU), continues serving on GPUs 4-7. Megatron trains on GPUs 0-3 with TP=2, CP=2.
3. **After training**: Updated weights are transferred to all 8 vLLM workers via CUDA IPC. vLLM expands back to all 8 GPUs.

This is why CUDA IPC permissions (Problem 1) are required -- the weight transfer in step 3 uses cross-process GPU memory access.

### Modified File: `ROLL-personal/examples/agentic_demo/agent_val_rock_swe_qwen35_08b.yaml`

Updated from Phase 4 with the same Mode B flags and 8-GPU allocation:

| Parameter | Phase 4 Value | Phase 5 Value | Reason |
|---|---|---|---|
| `async_generation_ratio` | (not set) | `1` | Enable Mode B |
| `parse_tool_call_parameter_to_dict` | (not set) | `true` | Jinja2 fix |
| `skip_mock_system_prompt` | (not set) | `true` | Template compat |
| `track_with` | (not set) | `wandb` | Logging |
| `num_gpus_per_node` | `8` | `8` | (unchanged) |
| `rollout_batch_size` | `1` | `4` | DP=4 requires divisible batch |
| `val_batch_size` | `1` | `4` | DP=4 requires divisible batch |
| `actor_train.device_mapping` | `list(range(0,4))` | `list(range(0,4))` | Train on GPUs 0-3 |
| `actor_train.strategy_config.TP` | `1` | `1` | (unchanged, TP=1 for 0.8B) |
| `actor_infer.device_mapping` | `list(range(0,8))` | `list(range(0,8))` | Infer on all 8 GPUs |
| `reference.device_mapping` | `list(range(0,4))` | `list(range(0,4))` | Reference on GPUs 0-3 |

Note: The 0.8B config keeps TP=1 for train/reference (model small enough), while the 2B config uses TP=2, CP=2.

---

## Execution Commands

### Qwen3.5-2B Pipeline
```bash
docker exec -it roll_swe_runner bash -c "
    export NCCL_NET_PLUGIN='' && export NCCL_TUNER_PLUGIN='' && export NCCL_NET=Socket && \
    cd /home/ubuntu/ALE-latest/ROLL-personal && \
    export PYTHONPATH='/home/ubuntu/ALE-latest/ROLL-personal:\$PYTHONPATH' && \
    python examples/start_agentic_pipeline.py \
        --config_path agentic_demo \
        --config_name agent_val_rock_swe_qwen35_2b
"
```

### Qwen3.5-0.8B Pipeline (Updated)
```bash
docker exec -it roll_swe_runner bash -c "
    export NCCL_NET_PLUGIN='' && export NCCL_TUNER_PLUGIN='' && export NCCL_NET=Socket && \
    cd /home/ubuntu/ALE-latest/ROLL-personal && \
    export PYTHONPATH='/home/ubuntu/ALE-latest/ROLL-personal:\$PYTHONPATH' && \
    python examples/start_agentic_pipeline.py \
        --config_path agentic_demo \
        --config_name agent_val_rock_swe_qwen35_08b
"
```

---

## Files Modified

| File | Change |
|---|---|
| `ROLL-personal/examples/agentic_demo/agent_val_rock_swe_qwen35_2b.yaml` | **NEW** -- 2B config with Mode B, TP=2, CP=2, 8-GPU allocation |
| `ROLL-personal/examples/agentic_demo/agent_val_rock_swe_qwen35_08b.yaml` | **Modified** -- added Mode B flags, Jinja2 fix, wandb, batch_size 1->4 |
| Docker container `roll_swe_runner` | **Recreated** -- added `--pid=host`, `--cap-add SYS_PTRACE`, `--security-opt seccomp=unconfined` |

---

## Results

### Jinja2 Fix Verified
- 0.8B pipeline ran 6+ successful environment steps with zero Jinja2 errors
- `parse_tool_call_parameter_to_dict: true` confirmed working

### Mode B (Partial GPU) Enabled
- Both 2B and 0.8B configs now use `async_generation_ratio: 1` for simultaneous train+infer
- vLLM inference spans all 8 GPUs; Megatron training uses GPUs 0-3
- CUDA IPC permissions resolved via container recreation

### Qwen3.5-2B Results
- Full Mode B cycle verified: Inference (8 GPUs) → Shrink (1.8s) → Train (9/9 micro-batches in 1:46) → Model Update → Expand
- Val trajectory: 15 steps, 7:23 min, avg 80.4 tokens/response at 32.6 tok/s
- Peak training memory: 20.7GB/40GB — comfortable
- No OOM, no Jinja2 errors
- WandB: https://wandb.ai/shamanework-pl/roll-agentic/runs/zaagjwts

### Qwen3.5-0.8B Results
- Jinja2 fix verified: 6+ env steps with zero template errors
- Mode B shrink/expand working
- **OOM during training at sequence_length=32768**: Megatron tried to allocate 30.31 GiB for activations during `forward_step` → `loss_func`. With TP=1, all activations on 1 GPU. Only 19.66 GiB free after vLLM shrink.
- **Root cause**: Long trajectories (up to 60 turns × ~2K tokens) concatenated into one sequence for training. With TP=1 and no CP, the full 32K context is processed on a single GPU.
- **Fix**: Reduced `sequence_length` from 32768 to 16384 and `gpu_memory_utilization` from 0.6 to 0.5
- **Why 2B didn't OOM**: TP=2 splits activation tensors across 2 GPUs, CP=2 halves context per GPU (32K/2=16K)
- WandB: https://wandb.ai/shamanework-pl/roll-agentic/runs/vurhlvia

---

## Problem 5: SWE-bench Dataset Registry Reference

### Symptom
Train env failed to pull Docker image `pallets__flask-5014` from Chinese Alibaba registry (`rex-registry.cn-hangzhou.cr.aliyuncs.com`) — unreachable from AWS with `dial tcp: i/o timeout`.

### Fix
Replaced Chinese registry prefix with Docker Hub in `ROLL-personal/data/swe_bench_verified_example.jsonl`. Pipeline auto-recovered after 3 failed retries by moving to next task.

---

## Comparison: 0.8B vs 2B Config

| Aspect | Qwen3.5-0.8B | Qwen3.5-2B |
|---|---|---|
| Model size | 0.8B params | 2B params |
| Train parallelism | TP=1, CP=1 | TP=2, CP=2, SP=true |
| Data parallelism | DP=4 | DP=1 |
| Batch size | 4 (divisible by DP=4) | 1 (DP=1, no constraint) |
| sequence_length | 16384 (reduced from 32K) | 32768 |
| gpu_memory_utilization | 0.5 | 0.6 |
| distributed_optimizer | true | true |
| recompute_granularity | full | full |
| Reference parallelism | TP=1, CP=1 | TP=2, CP=2 |
| OOM risk | High with long contexts (TP=1) | Low (CP=2 halves context) |

---

## Agent Tool Call Flow (Verified Working)

The 0.8B model correctly uses tools in the SWE-bench pipeline:

1. **Prompt**: System message includes 15 tool definitions (list_directory, read_file, search_file_content, glob, replace, write_file, run_shell_command, etc.) in JSON schema format inside `<tools>` tags
2. **Model generates**: Thinking text + `<tool_call><function=search_file_content><parameter=pattern>evalf.*_imp</parameter></function>`
3. **ACTION_PARSE**: Extracts tool name + parameters from XML-style tool call
4. **Execution**: iflow CLI executes the tool inside the SWE-bench Docker container via ROCK sandbox
5. **Observation**: Tool output returned as next turn's context
6. **Repeat**: Up to 60 steps per trajectory

Verified: Model uses `search_file_content`, `read_file` (with offset/limit), `replace`, `glob`, `run_shell_command` correctly across multiple tasks.

---

## Reward Normalization Issue (Known, Not Fixed)

With current config (`adv_estimator: step_reinforce`, `group_size: 1`, `reward_normalization.grouping: traj_group_id`):
- Each group has exactly 1 trajectory
- Normalization subtracts group mean from rewards
- Single-value group mean = the value itself → normalized reward = 0
- **Gradient signal is effectively zero**

Fix for future: Change `reward_normalization.grouping` to `batch` (normalize across all 4 trajectories) or increase `group_size` for within-prompt comparison (like GRPO).
