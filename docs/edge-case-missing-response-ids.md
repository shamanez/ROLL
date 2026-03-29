# Edge Case: History Entry with `prompt_ids` but no `response_ids`

## The Error

```
File "traj_env_manager.py", line 305, in formulate_rollouts
    token_ids.extend(items["response_ids"])
KeyError: 'response_ids'
```

## Timeline of a Normal Step

```
format_messages()          → sets prompt_ids on history[-1]
llm_proxy.generate()       → model generates tokens
make_decision()            → sets response_ids on history[-1]     (line 196)
step()                     → sets reward, llm_response on history[-1]  (line 151-152)
                           → appends NEW entry with only {observation, actions_left, messages}  (line 156-160)
```

The key insight: `prompt_ids` is set BEFORE the LLM call. `response_ids` is set AFTER.
If anything goes wrong between these two points, the entry has `prompt_ids` but no `response_ids`.

## All Code Paths That Produce This

### Path 1: Normal trajectory termination (MOST COMMON — this is what we hit)

After the final `step()` call (line 156-160), a new history entry is always appended:

```python
self.rollout_cache.history.append({
    "observation": copy.deepcopy(observation),
    "actions_left": self.env_config.max_steps - self.rollout_cache.step,
    "messages": None
})
```

Then at line 87: `if self.running and rollout_cache.terminated:` → calls `formulate_rollouts()`.

This trailing entry has `observation` only — no `prompt_ids`, no `response_ids`.
This is the **normal** case. Every terminated trajectory has this.
`formulate_rollouts` in step mode already handles it (`if "response_ids" not in history: break`).

**Verdict: NOT a bug. Normal bookkeeping. Not scary.**

### Path 2: LLM proxy returns None (ABORT)

```python
# make_decision(), line 182-187
lm_output = self.llm_proxy.generate(...)
if lm_output is None:
    return DataProto(meta_info={"stop_reason": GenerateStopReason.ABORT})
```

At this point, `format_messages()` already ran (line 164), which set `prompt_ids` on `history[-1]`.
But `response_ids` is never set (line 196 is skipped).

Back in `run_rollout_loop()`, line 83:
```python
if stop_reason in [GenerateStopReason.FINISH, GenerateStopReason.MAX_LENGTH]:
    rollout_cache = self.step(lm_output)
```
ABORT is not in that list → `step()` is never called → `reward` is also missing.

**When does this happen?**
- vLLM worker crashed or timed out
- Ray actor died mid-generation
- Network partition between env worker and inference worker

**Verdict: Error path. Entry has prompt_ids but no response_ids AND no reward.**

### Path 3: Sequence length exceeded (MAX_LENGTH before LLM call)

```python
# make_decision(), line 167-170
if input_ids.shape[1] >= self.pipeline_config.sequence_length:
    return DataProto(meta_info={"stop_reason": GenerateStopReason.MAX_LENGTH})
```

`format_messages()` already ran → `prompt_ids` is set.
LLM is never called → `response_ids` is never set.

But MAX_LENGTH IS in the step condition (line 83), so `step()` gets called with an
lm_output that has no batch data — this could produce a corrupted entry or crash inside
`step()`. However if `step()` succeeds, the entry would have `prompt_ids` + `reward` but
still no `response_ids`.

**When does this happen?**
- Long multi-turn trajectories where accumulated context exceeds 32768 tokens
- More likely with traj mode since all turns are in one sequence

**Verdict: Edge case. Prompt was too long for the model to generate any tokens.**

## What the Fix Does

The fix in `traj_env_manager.py` adds the same guard that step mode always had:

```python
for items in self.rollout_cache.history:
    if "response_ids" not in items:
        break  # skip this entry — no model output to train on
```

This is safe because:
- Entries without `response_ids` have no model-generated tokens → no training signal
- Entries without `response_ids` also lack `reward` (Path 1, 2) or have corrupted reward (Path 3)
- The `scores` list also filters: `if 'reward' in i and 'response_ids' in i`
- All preceding entries with `response_ids` are fully intact and used for training

## Is This Dangerous?

**No.** The most common trigger (Path 1) happens on literally every trajectory — it's the
normal trailing observation entry. Step mode has been silently skipping it since day one.

Path 2 (LLM abort) and Path 3 (sequence too long) are rare edge cases that also produce
no useful training data. Skipping them is correct — there are no response tokens to learn from.

The error only surfaced because `TrajEnvManager.formulate_rollouts` was written for
environments where this edge case doesn't occur, and our traj integration delegated to it
without adding the guard.
