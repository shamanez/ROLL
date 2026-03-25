# Phase 4: Qwen3.5-0.8B Support - Solution

## Config Created
**File:** `ROLL-personal/examples/agentic_demo/agent_val_rock_swe_qwen35_08b.yaml`

Copy of `agent_val_rock_swe.yaml` with these changes:

| Parameter | Qwen3-4B | Qwen3.5-0.8B | Reason |
|---|---|---|---|
| pretrain | Qwen3-4B-Instruct-2507 | Qwen3.5-0.8B | Smaller model, fits on 1 GPU |
| reward_pretrain | Qwen3-4B-Instruct-2507 | Qwen3.5-0.8B | Match pretrain |
| actor_train.model_args.flash_attn | fa2 | sdpa | Qwen3.5 reference uses sdpa |
| actor_train.model_args.attn_implementation | (implicit) | sdpa | Explicit for Qwen3.5 |
| actor_train.model_args.freeze_module_prefix | (not set) | vision_model | Qwen3.5 is multimodal; freeze vision for text-only SWE |
| actor_infer.model_args.flash_attn | fa2 | sdpa | Match train |
| actor_infer.model_args.attn_implementation | (implicit) | sdpa | Explicit |
| reference.model_args.attn_implementation | fa2 | sdpa | Match |
| reference.model_args.freeze_module_prefix | (not set) | vision_model | Freeze vision |

## Results
- vLLM: 28.51 GiB KV cache (vs 23.35 for 4B - much more headroom)
- Training GPU: only ~2.4 GiB used (easily fits)
- Both sandboxes started successfully
- Agent installed and running (iflow-cli)
- Train trajectory completed with reward computation
- **No OOM** - 0.8B model fits easily on single A100-40GB
- Template issue on second trajectory (Jinja2 `items` filter on non-dict tool call args) - needs investigation
