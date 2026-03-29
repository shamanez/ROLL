# Training Report: Qwen3.5-2B with IPA Chunk-Level Loss on EndlessTerminals

## Summary

We trained **Qwen3.5-2B** (2B parameter causal LM) on **kanishk/EndlessTerminals** (2,490 terminal-based coding tasks) using **IPA chunk-level loss** with trajectory-level optimization. The model improved significantly from the untrained baseline:

| Metric | Baseline (untrained) | Step 200 | Step 400 | Step 700 |
|--------|---------------------|----------|----------|----------|
| **Validation score** (n=32, fixed seed) | **21.9%** (7/32) | **34.4%** (11/32) | **50.0%** (16/32) | **31.2%** (10/32) |
| Train success rate (50-step avg) | 28.0% | 36.9% | 38.5% | 42.3% |
| Avg actions per task (val) | 22.0 | 16.2 | 10.8 | 14.6 |
| Avg actions per task (train) | 18.2 | 13.7 | 14.1 | 11.2 |
| Failed tool calls | 27.2% | ~2% | ~5% | ~2% |

All validation scores measured on the same 32 tasks with fixed seed for fair comparison. Baseline = untrained Qwen3.5-2B.

**Wandb**: project `roll-agentic`, runs [`kkspbsu4`](https://wandb.ai/shamanework-pl/roll-agentic/runs/kkspbsu4) (steps 0-199) and [`scrrezmd`](https://wandb.ai/shamanework-pl/roll-agentic/runs/scrrezmd) (steps 200-699)

---

## 1. Setup

- **Model**: Qwen3.5-2B, pretrained checkpoint at `/model-checkpoints/Qwen3.5-2B`
- **Environment**: kanishk/EndlessTerminals via OpenReward API
  - 2,490 train tasks, random sampling (no epochs)
  - Binary reward: 1 if all tests pass, 0 otherwise
  - Up to 25 agent turns per trajectory
- **Hardware**: 8x NVIDIA A100-SXM4-40GB
  - **Dedicated 4+4 GPU split**: GPUs 0-3 for Megatron training (TP=2, CP=2), GPUs 4-7 for vLLM inference (4 independent workers)
- **Training**: 700 steps, `rollout_batch_size=16`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=16`, LR=1e-6 with 10-step warmup then constant
- **Config**: `openreward_endless_terminals_IPA_qwen35_2b_v2.yaml`

---

## 2. IPA Chunk-Level Loss

IPA (**Interaction-Perceptive Agentic Policy Optimization**) from the ROME paper operates at the **interaction chunk** level. Each agent turn (reasoning + tool call) is one chunk. This is the natural decision boundary for agentic tasks: individual tokens don't trigger environment transitions, and full sequences are too coarse for multi-turn credit assignment.

### 2.1 Chunk-Level Discounted Return (Paper Eq. 7)

For a trajectory with K chunks and terminal reward R_final:

```
G_k = gamma^{K-k} * R_final     (gamma = 0.95)
```

All tokens within chunk k share the same scalar weight G_k. Later chunks (closer to the outcome) receive higher credit. With gamma=0.95 and 15 chunks, the first chunk gets `0.95^14 = 0.49` of the reward while the last gets the full reward. This provides meaningful temporal credit without the vanishing signal of token-level discounting over thousands of tokens.

**Implementation**: `compute_discounted_returns()` in `utils.py` detects trajectory mode via `isinstance(step_scores[0], list)`, computes backward discounts, then fills a token-level tensor using `response_mask` segment boundaries found via `torch.diff()`.

### 2.2 Two-Branch Loss (Paper Eq. 9)

IPA splits trajectories into positive (R > 0) and negative (R <= 0) branches:

- **Positive trajectories** (task solved): **Weighted SFT** with no importance sampling ratio:
  ```
  L_positive = -G_k * log pi(c_k | tau_{<k})
  ```
  Directly reinforces successful chunks proportional to their temporal credit. This is pure on-policy learning — the model is trained on its own successful behavior.

- **Negative trajectories** (task failed): **TIS (Truncated Importance Sampling)** clipped to [0, 1]:
  ```
  L_negative = -clip(rho_k, 0, 1) * G_k * log pi(c_k | tau_{<k})
  ```
  The clip to [0, 1] means the gradient can only reduce probability of failed actions, never amplify it. With `ipa_failure_reward: null` and binary rewards, G_k = 0 for failures, so **negative trajectories contribute zero gradient**. The model learns exclusively from success.

**Implementation**: `_compute_ipa_chunk_loss()` in `agentic_actor_pg_worker.py` (line 473). `get_episode_scores()` determines positive/negative split per trajectory. `positive_token_mask` broadcasts to all tokens.

### 2.3 Using vLLM Logprobs as Behavior Policy (Paper Eq. 8)

In async training, the model weights are updated between rollout generation and training. The **behavior policy** (the model that generated the trajectory) no longer exists — its weights have been overwritten. The only surviving record is the **vLLM inference logprobs** stored during generation.

We use these as the behavior policy for the IS ratio:

```python
# agentic_actor_pg_worker.py line 66-73
log_ratio = log_probs - infer_log_probs           # pi_current / pi_vllm
masked_log_ratio = compute_segment_masked_mean(log_ratio, response_mask)  # geometric mean per chunk
ratio = masked_log_ratio.exp()                     # chunk-level IS ratio
```

Key points:
- `log_probs` = current Megatron training policy (after gradient updates)
- `infer_log_probs` = vLLM rollout logprobs (the actual behavior policy)
- `compute_segment_masked_mean` averages `log_ratio` within each contiguous response segment independently, then exponentiates — giving the **geometric mean** of per-token IS ratios per chunk
- This is more stable than the product of token ratios (avoids extreme values) and more precise than sequence-level IS (each chunk gets its own correction)
- `force_disable_old_logprobs_recompute: true` ensures vLLM logprobs are used as-is, not recomputed by Megatron

### 2.4 Chunk-Level Mismatch Masking

Chunks with geometric mean IS ratio exceeding threshold H=5.0 are masked out entirely:

```yaml
train_infer_correction:
  filters:
    - enabled: true
      agg_type: segment          # chunk-level geometric mean
      ratio_high: 5.0            # threshold H
```

This is computed in `compute_train_infer_correction()` (`train_infer_corrections.py`), which zeros out `response_mask` for offending chunks **before** the loss computation. Masked chunks contribute zero gradient.

**Observed metrics**: `actor/train_infer_final_mask_mean = 1.0` throughout training — no chunks were masked. This means the async staleness was well-controlled (IS ratio stayed near 1.0).

---

## 3. Trajectory-Level Optimization (`formulate_mode: traj`)

### 3.1 The Problem with Step Mode

The default `AgentNativeStepEnvManager` creates **one training sample per chunk**. For a trajectory with K=15 turns:

| Issue | Impact |
|-------|--------|
| K forward/backward passes per trajectory | 15x more compute |
| K optimizer steps per pipeline step | LR scheduler advances 15x too fast |
| K weight updates before next rollout | Higher async staleness |
| K padded sequences of length 32768 | Wasted padding tokens |

With a cosine LR schedule over 200 pipeline steps and 15 turns/trajectory, the LR scheduler would exhaust in ~13 pipeline steps. With `constant_with_warmup`, the warmup completes in <1 pipeline step instead of the intended 10.

### 3.2 How Trajectory Mode Works

With `formulate_mode: "traj"`, all chunks are packed into a **single training sample**:

```
[prompt_1 | response_1 | prompt_2 | response_2 | ... | prompt_K | response_K | padding]
  mask=0     mask=1       mask=0     mask=1              mask=0     mask=1       mask=0
```

The `response_mask` has K contiguous segments of 1s. Each segment is one chunk. All existing IPA machinery works unchanged because `compute_segment_masked_mean` independently processes each segment:

```python
# For each contiguous run of 1s in response_mask:
#   1. Average log_ratio within that segment → chunk-level geometric mean IS ratio
#   2. Broadcast the mean back to all tokens in the segment
```

### 3.3 Benefits

| | Step mode | Traj mode |
|---|---|---|
| Forward/backward per trajectory | K (~15) | 1 |
| Optimizer steps per pipeline step | K | 1 |
| LR scheduler accuracy | K x too fast | Correct |
| Compute per trajectory | ~7x | 1x |
| Async staleness (weight updates between syncs) | K | 1 |

### 3.4 Modifications Required

Adapting `AgentNativeStepEnvManager` to support trajectory mode required changes at the boundary between `AgentNativeStepEnvManager` and `TrajEnvManager`:

1. **Dispatcher** in `formulate_rollouts()` — routes to `_formulate_rollouts_step` or `_formulate_rollouts_traj` based on config
2. **`_formulate_rollouts_traj`** — pre-computes metadata, delegates token assembly to `TrajEnvManager.formulate_rollouts()`, adds AgentNative-specific metadata
3. **Edge case guards** — 5 bugs fixed at the delegation boundary (see Section 6)
4. **`compute_discounted_returns`** — traj mode branch that fills token-level `step_rewards` using `response_mask` segment boundaries
5. **Defensive dimension handling** — ensures `step_rewards` is always 2D before batch concat

---

## 4. Async Pipeline

### 4.1 Dedicated 4+4 GPU Split

```
GPUs 0-3: Megatron training (actor_train + reference model)
           TP=2, CP=2, sequence_parallel=true, distributed_optimizer=true

GPUs 4-7: vLLM inference (4 independent workers)
           TP=1, gpu_memory_utilization=0.8, max_model_len=32768
```

The reference model shares GPUs 0-3 with training — it runs `megatron_infer` (read-only forward pass) between rollout collection and training, using the same model shards.

### 4.2 Async Generation (`async_generation_ratio: 64`)

With async mode, vLLM workers on GPUs 4-7 generate trajectories **continuously** — including during the training phase on GPUs 0-3. Generated trajectories are buffered in the rollout scheduler.

The pipeline step cycle:
```
1. offload_states (actor_train)              ~0s
2. model_update (actor_train → actor_infer)  ~3-5s (weight sync to vLLM)
3. load_states (actor_infer KV cache)        ~1s
4. get_batch (pull 16 from buffer)           ~0-200s (depends on buffer depth)
5. shrink_sampler (free GPUs 0-3 from vLLM)  ~1s
6. compute_discounted_returns                ~0s
7. reference.compute_log_probs               ~2s
8. actor_train.train_step (16 micro-batches) ~25-30s
```

With deep enough buffer, `get_batch` returns instantly (trajectories already generated). The main overhead is the weight sync (~3-5s) and training (~25-30s).

### 4.3 Staleness and IS Correction

Buffered trajectories were generated by older policy weights. The chunk-level IS ratio corrects for this:
- `actor/ratio_mean ≈ 0.99` — nearly on-policy throughout training
- `actor/train_infer_final_mask_mean = 1.0` — no chunks masked for staleness
- The geometric mean IS ratio per chunk is more stable than token-level products

---

## 5. Training Dynamics

### 5.1 Success Rate Over 700 Steps

```
Steps   0- 49:  28.0%  avg_actions=18.2  |=======             | (warmup)
Steps  50- 99:  33.3%  avg_actions=15.9  |=========           | (learning starts)
Steps 100-149:  36.0%  avg_actions=14.9  |==========          | (steady gain)
Steps 150-199:  36.9%  avg_actions=13.7  |==========          | (first plateau)
Steps 200-249:  40.9%  avg_actions=14.5  |===========         | (resumed from ckpt)
Steps 250-299:  38.6%  avg_actions=15.0  |==========          |
Steps 300-349:  37.9%  avg_actions=15.7  |==========          |
Steps 350-399:  37.3%  avg_actions=15.1  |==========          |
Steps 400-449:  38.5%  avg_actions=14.1  |==========          |
Steps 450-499:  38.5%  avg_actions=12.8  |==========          | (efficiency improving)
Steps 500-549:  38.8%  avg_actions=12.2  |==========          |
Steps 550-599:  36.9%  avg_actions=12.1  |==========          |
Steps 600-649:  37.2%  avg_actions=11.7  |==========          |
Steps 650-699:  42.3%  avg_actions=11.2  |============        | (new high)
```

### 5.2 Validation Results

#### Proper evaluation (n=32, fixed seed, apples-to-apples)

| Model | Val Score | Tasks Solved | Val Avg Actions |
|-------|-----------|-------------|-----------------|
| Baseline (untrained) | **21.9%** | 7/32 | 22.0 |
| Step 200 (checkpoint-199) | **34.4%** | 11/32 | 16.2 |
| Step 400 (checkpoint-400) | **50.0%** | 16/32 | 10.8 |
| Step 700 (checkpoint-699) | **31.2%** | 10/32 | 14.6 |

**Key insight**: Step 400 is the best checkpoint on validation (50%, +128% over baseline). Step 700 shows higher train success (42.3%) but lower val score (31.2%), suggesting overfitting to the training distribution after step 400. The optimal checkpoint for deployment is **step 400**.

#### During-training validation (n=4, high variance)

| Step | Val Score | Tasks Solved | Val Avg Actions |
|------|-----------|-------------|-----------------|
| 200 | 50.0% | 2/4 | 15.5 |
| 300 | 50.0% | 2/4 | 16.8 |
| 400 | 75.0% | 3/4 | 8.2 |
| 500 | 50.0% | 2/4 | 13.2 |
| 600 | 50.0% | 2/4 | 8.5 |

### 5.3 Training Health Metrics (Step 200)

| Metric | Value | Meaning |
|--------|-------|---------|
| `actor/ratio_mean` | 0.995 | IS ratio nearly 1.0 (on-policy) |
| `actor_train/grad_norm` | 0.988 | Stable, below clip threshold 1.0 |
| `actor/ipa_positive_ratio` | 0.25 | 4/16 positive trajectories per batch |
| `actor/train_infer_final_mask_mean` | 1.0 | No chunks masked (all within threshold) |
| `actor/ipa_chunk_ratio_mean` | 0.9998 | Minimal per-chunk policy drift |
| `actor/lr` | 1e-6 | Constant after warmup |
| `actor/pg_loss` | 0.056 | IPA loss magnitude |

---

## 6. Traj Mode Integration: Bugs Fixed

Adapting `AgentNativeStepEnvManager` to delegate to `TrajEnvManager` exposed 5 bugs at the boundary, all caused by differing assumptions about data format:

| Bug | File | Root Cause | Fix |
|-----|------|-----------|-----|
| `KeyError: 'response_ids'` | traj_env_manager.py | TrajEnvManager iterates history without checking for observation-only trailing entries | Added `if "response_ids" not in items: break` guard |
| `RuntimeError: got 2 and 1` | utils.py | `np.array([list])` unpacks list into 2D; after DataProto ops, traj mode detection fails | `_wrap_as_object_array()` helper preserves list-in-object-array |
| `RuntimeError: got 2 and 1` (intermittent) | utils.py | `adjust_batch("copy")` duplicates samples; group has batch_size>1 but `step_rewards` computed for 1 | `.expand(num_samples, -1)` matches batch size |
| `ValueError: 1 is not in list` | traj_env_manager.py | Empty trajectory has no response tokens; `response_masks.index(1)` crashes | `if 1 in response_masks else 0` fallback |
| `IndexError` on segment boundaries | utils.py | `num_segments` didn't account for `segment_ends` length | `min(segment_starts, segment_ends, discounts)` |

Additionally, a defensive 1D-to-2D force-fix ensures any trajectory that falls through to step mode gets its `step_rewards` reshaped before batch concatenation.

All fixes are in the runtime code path and have been validated over 700 training steps with zero crashes.

---

## 7. Checkpoints

| Step | Path | Notes |
|------|------|-------|
| 50 | `/data/20260327-220517/checkpoint-50` | Early training |
| 100 | `/data/20260327-220517/checkpoint-100` | |
| 150 | `/data/20260327-220517/checkpoint-150` | |
| 199 | `/data/20260327-220517/checkpoint-199` | End of first run |
| 250-699 | `/data/20260328-084000/checkpoint-*` | Continuation run |

---

## 8. Reproducibility

```bash
docker exec -it roll_openreward_runner bash
export NCCL_NET_PLUGIN='' && export NCCL_TUNER_PLUGIN='' && export NCCL_NET=Socket
cd /home/ubuntu/ALE-latest/ROLL-personal
export PYTHONPATH="$PWD:$PYTHONPATH"
export OPENREWARD_API_KEY="<key>"
export WANDB_API_KEY="<key>"

python examples/start_agentic_pipeline.py \
  --config_path agentic_demo \
  --config_name openreward_endless_terminals_IPA_qwen35_2b_v2
```

Config: `ROLL-personal/examples/agentic_demo/openreward_endless_terminals_IPA_qwen35_2b_v2.yaml`

**GitHub feature request**: [alibaba/ROLL#409](https://github.com/alibaba/ROLL/issues/409) — Trajectory-level formulation mode for AgentNativeStepEnvManager
