# Phase 3: Qwen3-4B Baseline - Solution

## What Was Done
Ran the Qwen3-4B pipeline with minimal config changes.

## Config Modification
**File:** `ROLL-personal/examples/agentic_demo/agent_val_rock_swe.yaml`

**Change:** Added `max_model_len: 32768` to `actor_infer.strategy_args.strategy_config`

**Reason:** Qwen3-4B's default max_seq_len is 262144 (256K), which requires 36 GiB KV cache - more than the available 23.35 GiB on a single A100-40GB. Setting `max_model_len: 32768` (matching `sequence_length`) reduces KV cache to fit in memory.

## Code Fix: Tokenizer return type (agent_native_env_manager.py)

**File:** `ROLL-personal/roll/pipeline/agentic/env_manager/agent_native_env_manager.py`

**Problem:** In transformers 5.2.0, `tokenizer.apply_chat_template(tokenize=True)` returns a `BatchEncoding` object instead of a plain `list[int]`. The code at line 206 does `torch.tensor(prompt_ids, dtype=torch.long)` which fails because a BatchEncoding can't be converted to a tensor directly.

**Fix:** Added `return_dict=False` parameter to `apply_chat_template()` call, which forces it to return a plain `list[int]`.

## Results
- vLLM initialized successfully (23.35 GiB KV cache, GPU 1)
- Megatron training loaded (GPU 2)
- Both sandboxes started successfully
- Train trajectory completed (60 agent steps, ~10 min)
- Val trajectory completed (reward computed)
- **OOM during `compute_log_probs`** - Qwen3-4B too large for single-GPU training with 32K context (needs 18.55 GiB more than available)

## Conclusion
Pipeline works end-to-end except for OOM at training step. The 4B model needs multi-GPU TP/PP to fit training on A100-40GB.
