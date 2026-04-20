# Complete Summary of All Modifications

## Overview

This document lists every file modification made during the ROLL+ROCK agentic SWE pipeline setup, with reasons.

---

## 1. ROCK Fixes (2 files modified)

### Fix 1: `ROCK-personal/rock/deployments/runtime_env.py`
**What:** Added symlink resolution for uv-managed Python paths in `LocalRuntimeEnv.get_volume_mounts()`

**Why:** ROCK mounts `.venv` into sandbox containers. The `.venv/bin/python` symlink chain goes through a uv symlink (`cpython-3.11-linux-x86_64-gnu` → `cpython-3.11.15-linux-x86_64-gnu`). Docker only mounts real directories, not symlinks. This caused the rocklet process inside sandbox containers to fail with "bad interpreter: No such file or directory", making ALL sandbox creation fail with a 600s timeout.

**Change:** Added `import os` and a block after the existing mount_configs that resolves the `.venv/bin/python` symlink chain. If the symlink target parent differs from the real path, it adds an additional volume mount mapping the real directory to the symlink path.

### Fix 2: `ROCK-personal/rock/sandbox/sandbox_manager.py`
**What:** Serialize `PhaseStatus` and `State` objects before Redis storage in `get_status()`

**Why:** `SandboxInfo` contains `PhaseStatus` (Pydantic model) and `State` (enum) objects. When `get_status()` tried to store this in Redis via `json_set()`, it failed with `TypeError: Object of type PhaseStatus is not JSON serializable`. This prevented sandbox status tracking after successful container creation.

**Change:** Added serialization block before `json_set()` call that converts `PhaseStatus` via `.to_dict()` and enum values via `.value`.

---

## 2. ROLL Config Changes (1 file modified, 1 file created)

### Modified: `ROLL-personal/examples/agentic_demo/agent_val_rock_swe.yaml`
**What:** Added `max_model_len: 32768` to `actor_infer.strategy_args.strategy_config`

**Why:** Qwen3-4B's model config specifies `max_position_embeddings: 262144` (256K). Without explicit `max_model_len`, vLLM tries to allocate KV cache for the full 256K context, requiring 36 GiB - more than the 23.35 GiB available on a single A100-40GB. This caused vLLM to fail with `ValueError: To serve at least one request with the model's max seq len (262144), 36.0 GiB KV cache is needed`. Setting `max_model_len: 32768` matches the config's `sequence_length` and fits in memory.

### Created: `ROLL-personal/examples/agentic_demo/agent_val_rock_swe_qwen35_08b.yaml`
**What:** New config for Qwen3.5-0.8B model

**Why:** Qwen3.5-0.8B (0.8B params) is small enough to run on a single A100-40GB for both inference and training. Changes from the 4B config:
- `pretrain`/`reward_pretrain`: points to `Qwen3.5-0.8B` checkpoint
- `flash_attn`/`attn_implementation`: `sdpa` instead of `fa2` (following Qwen3.5-27B reference config pattern)
- `freeze_module_prefix: vision_model`: Qwen3.5 is multimodal; freeze vision encoder for text-only SWE tasks
- `exp_name`: changed to `agentic_rollout_swe_qwen35_08b`

---

## 3. ROLL Code Fix (1 file modified)

### `ROLL-personal/roll/pipeline/agentic/env_manager/agent_native_env_manager.py`
**What:** Added `return_dict=False` parameter to `tokenizer.apply_chat_template()` call at line 203

**Why:** In transformers 5.2.0, `apply_chat_template(tokenize=True)` returns a `BatchEncoding` object instead of a plain `list[int]`. The subsequent `torch.tensor(prompt_ids, dtype=torch.long)` call fails with `TypeError: 'str' object cannot be interpreted as an integer` because `BatchEncoding` isn't directly convertible. Adding `return_dict=False` forces the method to return a plain `list[int]`.

---

## 4. Files Created (not modifications)

| File | Purpose |
|---|---|
| `Plans/01_docker_container_setup.md` | Phase 1 plan |
| `Plans/02_data_environment_prep.md` | Phase 2 plan |
| `Plans/03_run_qwen3_4b_baseline.md` | Phase 3 plan |
| `Plans/04_enable_qwen35_support.md` | Phase 4 plan |
| `Plans/05_run_qwen35_0.8b_pipeline.md` | Phase 5 plan |
| `solutions/install_up_roll.sh` | Reproducible ROLL container setup |
| `solutions/install_up_rock.sh` | Reproducible ROCK Admin setup |
| `solutions/phase1_docker_setup.md` | Phase 1 solution doc |
| `solutions/phase2_rock_and_data.md` | Phase 2 solution doc |
| `solutions/phase3_qwen3_4b_baseline.md` | Phase 3 solution doc |
| `solutions/phase4_qwen35_support.md` | Phase 4 solution doc |
| `solutions/phase5_qwen35_2b_pipeline.md` | Phase 5 solution doc |
| `Plans/06_run_qwen35_2b_pipeline.md` | Phase 6 plan |

---

## Pipeline Run Results

### Qwen3-4B (baseline)
- vLLM inference: OK (GPU 1, 23.35 GiB KV cache)
- Megatron training init: OK (GPU 2)
- ROCK sandboxes: OK (both train and val)
- Agent (iflow-cli) install: OK
- Full trajectory rollout: OK (60 agent steps)
- Reward computation: OK
- **Training step: OOM** - `compute_log_probs` needs 18.55 GiB more than available on single GPU

### Qwen3.5-0.8B (Phase 4)
- vLLM inference: OK (GPU 1, 28.51 GiB KV cache)
- Megatron training init: OK (GPU 2, only ~2.4 GiB)
- ROCK sandboxes: OK
- Agent install: OK
- Full train trajectory: OK (reward computed)
- **No OOM** - model fits easily
- Template issue on 2nd trajectory (Jinja2 `items` filter on non-dict tool args) - **FIXED in Phase 5**

---

## 5. Phase 5: Qwen3.5-2B Pipeline + Mode B + Jinja2 Fix (3 changes)

### Problem 1: CUDA IPC Permission Error
**What:** Mode B (partial GPU, simultaneous train+infer) requires CUDA IPC to transfer weights between Megatron train workers and vLLM infer workers on shared GPUs (0-3). CUDA IPC uses `pidfd_getfd` which is blocked by Docker's default seccomp profile.

**Fix:** Recreated `roll_swe_runner` container with added flags:
- `--pid=host` — share host PID namespace for cross-process FD access
- `--cap-add SYS_PTRACE` — allow `pidfd_getfd` syscall
- `--security-opt seccomp=unconfined` — disable seccomp filter

### Problem 2: Container Rebuild Lost Installed Packages
**What:** After recreating the container, ROLL, mcore_adapter, rl-rock, and terminal-bench-datasets were missing.

**Fix:** Re-ran install commands: `pip install -e . && pip install -e ./mcore_adapter && pip install rl-rock && pip install -r requirements_common.txt` and re-cloned terminal-bench-datasets.

### Problem 3: Batch Size / DP Validation (0.8B Config)
**What:** With TP=1 on 4 GPUs, DP=4. `rollout_batch_size: 1` is not divisible by 4.

**Fix:** Changed `rollout_batch_size` and `val_batch_size` from 1 to 4 in `agent_val_rock_swe_qwen35_08b.yaml`.

### Jinja2 Fix (Verified)
**What:** Qwen3.5 chat template uses Jinja2 `items()` on tool call args. Fails when args are JSON strings instead of dicts.

**Fix:** Set `parse_tool_call_parameter_to_dict: true` in both configs. Activates `json.loads()` conversion at `token_mask_utils.py:304-311`.

**Verification:** 0.8B pipeline ran 6+ successful environment steps with ZERO Jinja2 errors.

### Config: `agent_val_rock_swe_qwen35_2b.yaml` (NEW)
- Model: Qwen3.5-2B
- Mode B: `async_generation_ratio: 1` (simultaneous train+infer)
- Train: GPUs 0-3, Megatron, TP=2, CP=2, SP=true, distributed_optimizer=true
- Infer: GPUs 0-7, vLLM, TP=1, 8 DP workers, gpu_memory_utilization=0.6
- Reference: GPUs 0-3, megatron_infer, TP=2, CP=2
- Jinja2 fix: `parse_tool_call_parameter_to_dict: true`
- Template: `skip_mock_system_prompt: true`
- WandB: project `roll-agentic`

### Config: `agent_val_rock_swe_qwen35_08b.yaml` (Modified)
- Added: `async_generation_ratio: 1`, `parse_tool_call_parameter_to_dict: true`, `skip_mock_system_prompt: true`, wandb tracking
- Changed: `rollout_batch_size` and `val_batch_size` from 1 to 4
- GPU allocation: train GPUs 0-3, infer GPUs 0-7, reference GPUs 0-3

### Files Modified/Created in Phase 5
| File | Change |
|---|---|
| `ROLL-personal/examples/agentic_demo/agent_val_rock_swe_qwen35_2b.yaml` | NEW — 2B config with Mode B, TP=2, CP=2 |
| `ROLL-personal/examples/agentic_demo/agent_val_rock_swe_qwen35_08b.yaml` | Modified — Mode B, Jinja2 fix, wandb, batch_size 1->4 |
| Docker container `roll_swe_runner` | Recreated with --pid=host, SYS_PTRACE, seccomp=unconfined |
| `Plans/06_run_qwen35_2b_pipeline.md` | NEW — Phase 6 plan |
| `solutions/phase5_qwen35_2b_pipeline.md` | NEW — Phase 5 solution doc |

### Pipeline Results (Phase 5)

### Qwen3.5-0.8B (Mode B, 8 GPUs)
- Jinja2 fix verified: 6+ env steps with zero template errors
- Mode B enabled with `async_generation_ratio: 1`

### Qwen3.5-2B (Mode B, 8 GPUs)
- Config created: TP=2, CP=2 for train/reference, 8 vLLM workers for inference
- CUDA IPC permissions resolved via container recreation
- Full Mode B cycle verified: Inference (8 GPUs) → Shrink → Train (9/9 micro-batches, peak 20.7GB) → Model Update → Expand
- Val trajectory completed: 15 steps, 7:23 min, 80.4 avg tokens at 32.6 tok/s
- WandB: https://wandb.ai/shamanework-pl/roll-agentic/runs/zaagjwts

### Qwen3.5-0.8B (Mode B, 8 GPUs) — Overnight Run
- Jinja2 fix verified: 6+ env steps with zero template errors
- Mode B working: shrink/expand cycle confirmed
- **OOM during training**: `sequence_length: 32768` with TP=1 caused 30.31 GiB activation allocation on single GPU during `forward_step`. Only 19.66 GiB free after vLLM shrink.
- **Fix**: Reduced `sequence_length` from 32768 to 16384 and `gpu_memory_utilization` from 0.6 to 0.5
- **Root cause**: Long trajectories (up to 60 turns × ~2K tokens) fill the full sequence_length. With TP=1, all activations on 1 GPU. The 2B config avoids this via CP=2 (splits context across GPUs).
- WandB: https://wandb.ai/shamanework-pl/roll-agentic/runs/vurhlvia

---

## 6. Phase 5 Data Fix

### Dataset: `ROLL-personal/data/swe_bench_verified_example.jsonl`
**What:** One task (`pallets__flask-5014`) referenced a Chinese Alibaba registry (`rex-registry.cn-hangzhou.cr.aliyuncs.com/slimshetty/swebench-verified:...`) unreachable from AWS.

**Fix:** Replaced with Docker Hub path (`slimshetty/swebench-verified:sweb.eval.x86_64.pallets__flask-5014`). Pipeline auto-recovered after 3 failed retries by moving to next task.

### Install script: `solutions/install_up_roll.sh`
**What:** Updated docker run command with CUDA IPC flags (`--pid=host`, `--cap-add SYS_PTRACE`, `--security-opt seccomp=unconfined`).

### Solution doc: `solutions/phase1_docker_setup.md`
**What:** Updated docker run command with same CUDA IPC flags and explanation comment.

---

## 7. Known Issues / Configuration Notes

### Reward Normalization with group_size=1
With `adv_estimator: step_reinforce`, `group_size: 1`, and `reward_normalization.grouping: traj_group_id`: each group has 1 trajectory, so normalized rewards become zeros (mean of single value = value, value - mean = 0). For meaningful gradient signal, either increase `group_size` or change `grouping` to `batch`.

### SWE-bench Task Structure
Each of the 10 tasks has its own Docker image with the buggy repo pre-checked-out. Reward = 1 if the agent's code fix passes the test suite, 0 otherwise. With `rollout_batch_size: 4`, the pipeline picks 4 random tasks per training step.

---

## 8. Phase 6: GRPO (group_size=4) Overnight Run

### Problem: Zero Loss Despite group_size=4
With `reward_normalization.grouping: traj_group_id`, normalization is **per-trajectory** (each trajectory is its own group). Even with 4 trajectories per prompt, each trajectory's rewards are normalized against its own mean, producing zeros when all steps have equal reward. Loss only appears when a trajectory has **mixed step-level rewards** (within-trajectory variance).

### Config Changes (`agent_val_rock_swe_qwen35_2b.yaml`)
| Parameter | Before | After | Reason |
|---|---|---|---|
| `rollout_batch_size` | 1 | 4 | Must be divisible by group_size |
| `val_batch_size` | 1 | 4 | Same constraint |
| `group_size` (train+val) | 1 | 4 | GRPO: 4 trajectories per prompt |
| `gradient_accumulation_steps` | 1 | 4 | Micro batch=1, prevents OOM |

### Results (81 steps / 9.5 hours)
- Zero crashes, zero OOM, Mode B working throughout
- Checkpoint saved at step 50
- ~15/81 steps had non-zero scores (~18.5%)
- ~8/81 steps had real gradient updates (grad_norm 3.3-4.6)
- Best scores: 0.75 (3/4 trajectories solved) at steps 17, 24, 75
- WandB: https://wandb.ai/shamanework-pl/roll-agentic/runs/55s0tngc

### Key Learnings Documented
1. **Templates**: `parse_tool_call_parameter_to_dict: true` + `skip_mock_system_prompt: true` required for Qwen3.5
2. **OOM prevention**: `gradient_accumulation_steps` matches `group_size`; `gpu_memory_utilization: 0.6` for Mode B; CP=2 halves context per GPU
3. **Reward normalization**: `grouping: traj_group_id` normalizes within each trajectory, not across group. For true GRPO cross-trajectory comparison, change to `grouping: batch`
4. **Docker**: `--pid=host`, `--cap-add SYS_PTRACE`, `--security-opt seccomp=unconfined` required for CUDA IPC in Mode B
5. **Multi-turn loop**: anti-call pattern in `agent_native_env_manager.py:37-95` → `sandbox_manager_v2.py:1333-1399`

### Files
| File | Change |
|---|---|
| `ROLL-personal/examples/agentic_demo/agent_val_rock_swe_qwen35_2b.yaml` | group_size 1→4, batch sizes 1→4, grad_accum 1→4 |
| `solutions/phase6_grpo_overnight_run.md` | NEW — detailed Phase 6 doc |
