# Phase 6: GRPO (group_size=4) Overnight Run - Solution

## Overview

This phase fixes the zero-gradient problem from Phase 5 by enabling GRPO-style training (group_size=4) and documents all critical learnings about templates, OOM, advantage computation, and the multi-turn agentic loop.

---

## Problem 1: Zero Loss with group_size=1

### Symptom

Pipeline ran successfully but `actor/pg_loss@sum: 0.0` and `actor_train/grad_norm: 0.0` at every step. No learning happening despite trajectories completing.

### Root Cause

With `group_size: 1`, each prompt generates only 1 trajectory. The reward normalization (`grouping: traj_group_id`, `method: mean`) subtracts the group mean from each trajectory's reward. With 1 trajectory per group, the mean equals the value itself, so `normalized = value - mean(value) = 0` always.

Even with `group_size: 4` (GRPO), the normalization is **per-trajectory** (`traj_group_id` = unique per trajectory), not **across-group**. This means each trajectory's step rewards are normalized against their own mean within that trajectory — NOT compared against other trajectories in the group.

### Why Loss IS Sometimes Non-Zero

When a trajectory has **mixed step-level rewards** (some steps scored 0, some scored 1), within-trajectory normalization produces non-zero values. Steps with reward=1 get positive advantage, steps with reward=0 get negative. This creates gradient signal.

When all steps in a trajectory have the same reward (all 0 or all 1), normalization zeros everything out.

### Fix Applied

Changed `group_size: 1` → `4` plus supporting changes:

```yaml
# BEFORE (Phase 5)
rollout_batch_size: 1
val_batch_size: 1
group_size: 1  # (in both train_env_manager and val_env_manager)
gradient_accumulation_steps: 1

# AFTER (Phase 6)
rollout_batch_size: 4          # Must be divisible by group_size
val_batch_size: 4              # Same constraint
group_size: 4                  # GRPO: 4 trajectories per prompt
gradient_accumulation_steps: 4  # Micro batch = 1, accumulate over 4
```

### Why These Values

- `rollout_batch_size % group_size == 0` is enforced at `agentic_config.py:312-318`
- `gradient_accumulation_steps: 4` keeps micro batch = 1 (1 trajectory per forward pass), accumulates over 4 trajectories before optimizer step. This prevents OOM from processing all 4 at once.
- With DP=1 (TP=2, CP=2 on 4 GPUs): `actor_train_train_bsz = per_device_train_batch_size × gradient_accumulation_steps × dp_size = 1 × 4 × 1 = 4`

### Ideal Future Fix

Change `reward_normalization.grouping` from `traj_group_id` to `batch` or a custom grouping that puts all 4 trajectories from the same prompt into one normalization group. This would give: `[1,0,0,0] → normalized = [0.75, -0.25, -0.25, -0.25]` — proper cross-trajectory comparison.

---

## Problem 2: How Samples Are Generated (GRPO Architecture)

### Per Training Step

With `rollout_batch_size: 4`, `group_size: 4`, `num_env_groups: 1`:

| Metric | Value |
|---|---|
| Unique prompts from dataset | 1 |
| Parallel ROCK sandbox environments | 4 |
| Trajectories collected | 4 (one per env, same prompt) |
| Max steps per trajectory | 25 (`max_actions_per_traj`) |
| vLLM `num_return_sequences` | 1 (forced to 1 at `agentic_config.py:250-256`) |

### How 4 Trajectories Happen

NOT via vLLM multi-sampling. Instead:
1. **4 EnvironmentWorker** instances (train_env-0 through train_env-3) run in parallel
2. Each gets the same prompt with the same group seed
3. Each independently calls vLLM for generation (temperature=1.0 → different responses)
4. Each executes in its own ROCK sandbox container
5. **GroupQueue** (`rollout_scheduler.py:326`) waits for all 4 to complete before releasing for training

### Validation Constraints

```python
# agentic_config.py:309-318
assert rollout_batch_size % group_size == 0  # 4 % 4 = 0 ✓
assert val_batch_size % val_env_manager.group_size == 0  # 4 % 4 = 0 ✓
```

---

## Problem 3: Advantage Computation Pipeline

### Full Flow: Raw Reward → Gradient

```
1. SWE-bench binary reward (0/1 per step)
   ↓
2. Discounted returns (gamma=1.0, backward pass)
   utils.py:61-95 compute_discounted_returns()
   ↓
3. Reward normalization (per traj_group_id, subtract mean)
   utils.py:100-153 agentic_reward_norm()
   ↓
4. Token-level expansion (reward placed at EOS token)
   functionals.py:538-558 expand_to_token_level()
   ↓
5. REINFORCE returns (cumulative future rewards backward)
   functionals.py:485-496 compute_reinforce_return()
   ↓
6. Whitening (zero-mean, unit-variance)
   functionals.py:363-369 masked_whiten()
   ↓
7. Advantage clipping (±0.2)
   ↓
8. Policy gradient loss: -advantage × log_prob_ratio
```

### Why rollout/score Can Be Non-Zero But critic/rewards Is Zero

- `rollout/score/mean` = raw episode scores (pre-normalization), averaged across trajectories
- `critic/rewards/mean` = post-normalization step-level rewards
- Normalization subtracts per-trajectory mean, so `mean(normalized) = 0` by construction
- When max/min are also 0, it means ALL step rewards within each trajectory were identical (all 0s or all 1s)

---

## Critical Learnings: Templates

### Jinja2 `items()` Fix

**Problem:** Qwen3.5 chat template uses `{% for key, value in argument.items() %}` in Jinja2. When tool call arguments are JSON strings instead of Python dicts, `items()` fails.

**Fix:** `parse_tool_call_parameter_to_dict: true` in config. Activates `json.loads()` at `token_mask_utils.py:304-311` to convert string → dict before template rendering.

### Mock System Prompt Fix

**Problem:** Qwen3.5 chat template does NOT auto-add system prompt if missing. ROLL's default behavior adds a mock system prompt, which breaks the template.

**Fix:** `skip_mock_system_prompt: true` in config. Docstring at `agentic_config.py:229-231` explicitly says this is for Qwen3.5 series.

### Template Name

Use `template: qwen3_coder` for Qwen3.5 models (not `qwen3` or `qwen25`).

---

## Critical Learnings: OOM Errors

### 0.8B Model OOM (Phase 5)

**Symptom:** OOM during `forward_step` at `sequence_length: 32768` with TP=1
**Root cause:** TP=1 means all activations on 1 GPU. 30.31 GiB needed, only 19.66 GiB free after vLLM shrink.
**Fix:** Reduced `sequence_length: 16384` and `gpu_memory_utilization: 0.5`

### 2B Model: No OOM

**Why:** TP=2 splits model across 2 GPUs, CP=2 halves context per GPU (32K/2=16K effective). Peak training memory: 20.7GB/40GB — comfortable.

### OOM Prevention Checklist

| Parameter | Purpose |
|---|---|
| `gpu_memory_utilization: 0.6` | Leave room for Megatron on shared GPUs (Mode B) |
| `recompute_granularity: full` | Recompute activations instead of storing |
| `gradient_accumulation_steps: 4` | Process 1 trajectory at a time, not all 4 |
| `sequence_length` | Must fit in GPU memory with TP/CP. 32K OK for 2B with CP=2 |
| `max_model_len` | vLLM KV cache limit. Must match `sequence_length` |

### CUDA IPC for Mode B

Mode B (partial GPU, simultaneous train+infer) requires CUDA IPC for weight transfer between Megatron and vLLM on shared GPUs. Docker needs:
- `--pid=host` — share host PID namespace
- `--cap-add SYS_PTRACE` — allow `pidfd_getfd` syscall
- `--security-opt seccomp=unconfined` — disable seccomp filter

---

## Critical Learnings: Multi-Turn Agent Loop

### Architecture

```
ROLL (AgentNativeStepEnvManager)        ROCK (Sandbox + iflow-cli)
────────────────────────────────        ──────────────────────────
run_rollout_loop() [line 37-95]
  │
  ├─ make_decision() [line 160-196]
  │   └─ llm_proxy.generate()          → vLLM inference on GPU
  │
  ├─ step() [line 81]
  │   └─ env.step(action) [line 139]
  │       └─ sandbox_manager.format_response_payload()  [v2.py:1333]
  │       └─ sandbox_manager.fetch_agent_request()      [v2.py:1392]
  │           └─ agent_manager.anti_call_llm()          [agent_manager.py:140]
  │               └─ HTTP POST to Rocklet ──────────→   iflow-cli executes tool
  │                                                      Returns observation
  │               ←─────────────────────────────────
  │       └─ sandbox_manager.get_messages_and_tools()   [v2.py:1303]
  │
  └─ Repeat up to max_actions_per_traj (25) times
```

### Action Parsing

`action_parser.py:25-151` (Qwen3CoderActionParser) extracts tool calls via regex:
- Format 1: `<function=name><parameter=key>value</parameter></function>`
- Format 2: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`

### Anti-Call Pattern

ROLL pretends to be the LLM API. iflow-cli (inside ROCK container) sends requests to "localhost:8080/v1/chat/completions". ROLL intercepts these via the ModelService proxy, generates a response with vLLM, and sends it back. This is the "anti-call" pattern — the sandbox agent thinks it's calling an API, but ROLL is controlling the responses.

---

## Config Changes (Phase 6)

**File:** `ROLL-personal/examples/agentic_demo/agent_val_rock_swe_qwen35_2b.yaml`

| Parameter | Phase 5 | Phase 6 | Reason |
|---|---|---|---|
| `rollout_batch_size` | 1 | **4** | Must be divisible by group_size |
| `val_batch_size` | 1 | **4** | Same constraint |
| `group_size` (train) | 1 | **4** | GRPO: 4 trajectories per prompt |
| `group_size` (val) | 1 | **4** | Consistent |
| `gradient_accumulation_steps` | 1 | **4** | Keep micro_batch=1, prevent OOM |

Other parameters unchanged from Phase 5 (already correct):
- `async_generation_ratio: 1` (Mode B)
- `parse_tool_call_parameter_to_dict: true` (Jinja2 fix)
- `skip_mock_system_prompt: true` (Qwen3.5 template)
- `max_actions_per_traj: 25`
- `max_tokens_per_step: 4096`
- `sequence_length: 32768`
- `eval_steps: 0` (skip validation)
- `rollout_dump_dir: /home/ubuntu/ALE-latest/ROLL-personal/output/rollout_dump` (absolute path)

---

## Execution Command

```bash
docker exec -d roll_swe_runner bash -c "
    export NCCL_NET_PLUGIN='' && export NCCL_TUNER_PLUGIN='' && export NCCL_NET=Socket && \
    export WANDB_API_KEY='<key>' && \
    cd /home/ubuntu/ALE-latest/ROLL-personal && \
    export PYTHONPATH='/home/ubuntu/ALE-latest/ROLL-personal:\$PYTHONPATH' && \
    python examples/start_agentic_pipeline.py \
        --config_path agentic_demo \
        --config_name agent_val_rock_swe_qwen35_2b \
        > /home/ubuntu/ALE-latest/output_2b_grpo.log 2>&1
"
```

---

## Results (81 steps / 9.5 hours)

### Pipeline Health
- Zero crashes, zero OOM errors
- Mode B shrink/expand working throughout
- Checkpoint saved at step 50
- ~7 min/step average

### Training Signal
- **Steps with non-zero score:** ~15 out of 81 (~18.5%)
- **Steps with non-zero pg_loss:** ~8 out of 81 (~10%)
- **Typical grad_norm when non-zero:** 3.3 - 4.6
- **Score distribution:** 0.25 (1/4 solved), 0.5 (2/4), 0.75 (3/4)
- **Best step:** Step 17 (score=0.75, 3/4 trajectories solved)

### Score Timeline (non-zero steps only)
| Step | Score | Loss | Grad Norm |
|---|---|---|---|
| 5 | 0.25 | 0.0 | 0.0 |
| 7 | 0.25 | 0.10 | 3.34 |
| 17 | 0.75 | 0.0 | 0.0 |
| 18 | 0.50 | -0.016 | 4.57 |
| 19 | 0.50 | 0.0002 | 4.00 |
| 21 | 0.25 | 0.086 | 4.55 |
| 22 | 0.25 | 0.10 | 4.34 |
| 24 | 0.75 | -0.10 | 3.88 |
| 38 | 0.25 | 0.053 | N/A |
| 42 | 0.25 | 0.075 | N/A |
| 47 | 0.25 | 0.10 | N/A |
| 52 | 0.50 | 0.0 | 0.0 |
| 62 | 0.25 | 0.0 | 0.0 |
| 75 | 0.75 | 0.0 | 0.0 |
| 76 | 0.25 | 0.10 | N/A |
| 77 | 0.33 | 0.0 | 0.0 |

### WandB
- Run: `https://wandb.ai/shamanework-pl/roll-agentic/runs/55s0tngc`

---

## Files Modified in Phase 6

| File | Change |
|---|---|
| `ROLL-personal/examples/agentic_demo/agent_val_rock_swe_qwen35_2b.yaml` | group_size 1→4, rollout/val batch 1→4, gradient_accumulation 1→4 |
| `solutions/phase6_grpo_overnight_run.md` | NEW — this document |
| `solutions/SUMMARY_ALL_MODIFICATIONS.md` | Updated with Phase 6 changes |
