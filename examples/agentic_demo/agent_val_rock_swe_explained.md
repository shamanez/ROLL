# `agent_val_rock_swe.yaml` Deep Explanation

This note explains the exact behavior of:

- `ROLL-personal/examples/agentic_demo/agent_val_rock_swe.yaml`

It focuses on:

- what `ROLL`, `ROCK`, and `IFLOW` each do
- what `anti_call_llm` actually is
- the exact communication protocol around `anti_call_llm`
- whether this setup is multi-turn
- what the reward is in this example
- how reward becomes training loss
- one important consequence of this exact YAML: the effective learning signal can collapse to zero

## 1. The Three Main Pieces

### `ROLL`

`ROLL` is the RL trainer/orchestrator.

It does all of this:

- samples trajectories
- asks the policy model for the next assistant response
- stores each turn as a training sample
- computes returns / normalized rewards / advantages
- computes the actor loss and updates the policy

Relevant code:

- `roll/pipeline/agentic/agentic_pipeline.py`
- `roll/pipeline/agentic/env_manager/agent_native_env_manager.py`

### `ROCK`

`ROCK` is the sandbox and agent runtime layer.

It does all of this:

- starts the sandbox container for the SWE task
- creates the agent shell session
- installs and manages the agent runtime
- starts the local model-service bridge
- watches the agent process
- runs the task tests at the end

Relevant code:

- `roll/pipeline/agentic/env/sandbox/rock_tb_native_env.py`
- `roll/pipeline/agentic/env/rock/sandbox_manager_v2.py`
- `ROCK-personal/rock/sdk/model/*`

### `IFLOW`

`IFLOW` is the agent process being launched inside the sandbox.

In this YAML, the command is:

```yaml
run_cmd: 'iflow -p <<PROMPT>> --yolo'
```

Important subtlety:

- this example uses `agent_type: "default"`
- in ROCK, `"default"` maps to `RockAgent`
- so this is **not** the specialized ROCK `IFlowCli` class
- instead, it is `RockAgent` launching a generic shell command that happens to be `iflow`

So the agent runtime is:

```text
RockAgent  ->  runs shell command  ->  iflow -p ...
```

## 2. Two Different `localhost:8080` Meanings

The YAML contains both:

- `sandbox_base_url: http://localhost:8080`
- `IFLOW_baseUrl: "http://localhost:8080/v1"`

These look similar, but they are used in different contexts.

### From the ROLL process

`sandbox_base_url` is the ROCK control-plane address used by ROLL to create / manage sandboxes.

### From inside the sandbox

`IFLOW_baseUrl` is what the `iflow` agent uses as its OpenAI-compatible chat endpoint.

In this example it points to the local ROCK model-service running inside the sandbox:

```text
http://localhost:8080/v1/chat/completions
```

So:

```text
ROLL process perspective:
  sandbox_base_url -> ROCK sandbox service

inside sandbox perspective:
  IFLOW_baseUrl -> local model-service bridge
```

## 3. High-Level Runtime Sketch

```text
Outside sandbox
---------------
ROLL trainer
  |
  | asks ROCK to start task sandbox
  v
ROCK control plane


Inside sandbox
--------------
iflow agent  <->  local model-service (:8080/v1)
                     |
                     | local-mode anti_call_llm bridge
                     v
                 ROLL rollout loop
```

More concretely:

```text
ROLL env manager
  -> gets current messages/tools from ROCK
  -> sends them to policy model
  -> receives assistant response text
  -> sends that text back into ROCK/IFLOW
  -> IFLOW executes tools in sandbox
  -> IFLOW asks for next LLM call
  -> repeat
```

## 4. What `anti_call_llm` Means

`anti_call_llm` is the core bridge between:

- the agent inside the sandbox
- and the external controller that wants to provide the model outputs manually

The ROCK docs describe it like this:

- input: the previous LLM response
- output: the next LLM request

That is exactly how it is used here.

### Why it exists

Normally an agent would call the model directly.

With `anti_call_llm`, the direction is inverted:

1. the agent writes "here is the request I want to send to the LLM"
2. ROLL reads that request
3. ROLL runs its own policy model
4. ROLL sends back the model response
5. the agent continues

So `anti_call_llm` is the handshake that lets ROCK/IFLOW pause before every model call and let ROLL answer it.

## 5. `anti_call_llm` Protocol

In ROCK local mode, the communication medium is a log file.

### File markers

ROCK defines these markers:

```text
LLM_REQUEST_START
LLM_REQUEST_END
LLM_RESPONSE_START
LLM_RESPONSE_END
SESSION_END
```

### Request line format

```text
LLM_REQUEST_START{request_json}LLM_REQUEST_END{meta_json}
```

### Response line format

```text
LLM_RESPONSE_START{response_json}LLM_RESPONSE_END{meta_json}
```

### Session end

```text
SESSION_END
```

### Metadata

Both request and response metadata contain at least:

```json
{
  "timestamp": 1234567890,
  "index": 1
}
```

The `index` is the turn index for the model-service request/response pair.

## 6. Exact `anti_call_llm` Index Semantics

This part is easy to misunderstand.

The ROCK client logic is:

- `index = 0`, `last_response = None`
  - wait for the first request
  - return request with index `1`
- `index = n > 0`, `last_response = ...`
  - write response for request `n`
  - then wait for request `n + 1`

So it behaves like this:

```text
anti_call_llm(0, None)       -> request #1
anti_call_llm(1, response#1) -> request #2
anti_call_llm(2, response#2) -> request #3
...
```

That is why ROLL does this:

- on reset: fetch request with `index=0`
- on each later step: send response with current step index

## 7. What the Request Payload Looks Like

The request payload coming from IFLOW to model-service is an OpenAI-compatible chat request body.

Typical shape:

```json
{
  "model": "...",
  "messages": [...],
  "tools": [...]
}
```

In this ROLL integration, ROLL only needs a subset of that request:

- `messages`
- `tools`

That is why `get_messages_and_tools()` simply parses the JSON and extracts those two fields.

## 8. What the Response Payload Looks Like

ROLL does **not** send raw assistant text back directly.

It wraps the text into an OpenAI-compatible response payload.

Shape:

```json
{
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "...",
        "tool_calls": [...]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

If there are no tool calls:

```json
{
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "final answer"
      },
      "finish_reason": "stop"
    }
  ]
}
```

## 9. How Tool Calls Are Encoded

The model in this example emits tool calls in text form, for example with:

- `<tool_call>...</tool_call>`
- or `<function=...>...</function>`

ROLL parses those text blocks and converts them into OpenAI-style `tool_calls`.

Important consequence:

- one model response may contain zero, one, or multiple tool calls
- but ROLL still treats that whole response as **one turn**

So the training granularity is **not per tool call**.

It is per assistant response turn.

## 10. Exact End-to-End Turn Flow

Here is the exact handshake.

### Reset

```text
1. ROLL reset()
2. ROCK starts sandbox
3. ROCK starts RockAgent
4. RockAgent launches: iflow -p <task prompt> --yolo
5. IFLOW tries to call /v1/chat/completions
6. local model-service writes request #1 into the log file
7. ROLL calls anti_call_llm(index=0)
8. ROLL receives request #1 as JSON string
9. ROLL extracts messages/tools and builds model input
```

### One normal step

```text
1. ROLL policy model generates one assistant response
2. Response may contain tool calls or may be plain text
3. ROLL wraps it as OpenAI-style response payload
4. ROLL calls anti_call_llm(index=n, response_payload=response_n)
5. model-service writes response #n into the log file
6. IFLOW reads it, executes tools if needed
7. IFLOW eventually makes next model request
8. model-service writes request #(n+1)
9. anti_call_llm returns request #(n+1) to ROLL
```

### Termination

```text
1. agent process exits
2. watch_agent notices the PID exited
3. model-service writes SESSION_END
4. ROLL sees SESSION_END
5. environment computes final test reward
```

## 11. Is This Multi-Turn?

Yes.

Very clearly yes.

In this example:

- `max_actions_per_traj: 60`
- each action is one assistant response turn
- conversation history is accumulated across turns
- the next prompt is built from the full message history

So this is a multi-turn agent trajectory.

But be precise about the unit:

- **one turn** = one assistant response from the model
- that response may include multiple tool calls
- therefore **one turn is not the same thing as one tool call**

## 12. What Counts as One Training Sample?

This is the most important granularity point.

The env manager stores **one sample per assistant response step**.

Not per tool call.
Not per token chunk.
Not per environment sub-event.

Formally:

```text
sample_t = {
  prompt/history up to turn t,
  assistant response at turn t,
  step reward for turn t
}
```

If one assistant response contains two tool calls, that is still one sample.

## 13. What Is the Reward in This Exact SWE Example?

The reward source is the environment test result.

At the end of the trajectory, ROCK runs the task tests:

- uploads the task `tests/`
- uploads `run-tests.sh`
- runs the tests in the sandbox
- parses the output
- returns `is_resolved`

So in this exact environment, the reward is basically:

```text
reward = 1 if the task is resolved
reward = 0 otherwise
```

### Important detail

For most of the trajectory, the reward is still `0`.

The reward becomes non-zero only when the episode ends and tests are run.

So the **reward source is sparse / terminal-like**.

## 14. Step Rewards in This Env

Because reward is computed only at termination, the per-step reward sequence looks like:

### Success example

```text
step_scores = [0, 0, 0, 1]
```

### Failure example

```text
step_scores = [0, 0, 0, 0]
```

That is what gets stored in `history["reward"]` turn by turn.

## 15. Then What Does `compute_discounted_returns()` Do?

This is the part you asked about directly.

It computes:

```text
G_t = r_t + gamma * G_{t+1}
```

over the **step rewards across the trajectory**.

So yes:

- the original reward source is sparse / terminal-like
- but then ROLL converts it into a step-wise return for every turn

This means the terminal outcome is propagated backward across earlier turns.

### For this YAML

`step_reward_gamma: 1.0`

So for:

```text
step_scores = [0, 0, 0, 1]
```

the discounted step returns become:

```text
step_returns = [1, 1, 1, 1]
```

For failure:

```text
step_scores = [0, 0, 0, 0]
step_returns = [0, 0, 0, 0]
```

If this YAML had used `step_reward_gamma = 0.95` instead, then a terminal-only
success trajectory of length 4 would become approximately:

```text
step_scores  = [0,    0,    0,   1]
step_returns = [0.95^3, 0.95^2, 0.95, 1]
             = [0.8574, 0.9025, 0.95, 1]
```

That non-constant shape is exactly why `gamma = 1.0` matters so much in this file.

### So are you correct?

Your statement is close, but it needs one correction.

Correct version:

- yes, the terminal outcome is propagated backward to earlier turns
- no, the unit is **not each tool call chunk**
- the unit is **each assistant response turn**

So the right mental model is:

```text
trajectory
  -> split into assistant turns
  -> each turn gets a backward discounted return
```

Not:

```text
trajectory
  -> split into tool calls
  -> each tool call gets its own return
```

That second statement is not what this code does.

## 16. One More Important Correction: Not Cross-Turn Token Credit

After step returns are computed, each turn sample still gets only **one scalar response-level reward**.

That scalar is then attached to the **last token of that turn's response**.

Then REINFORCE computes backward token returns **within that single response**.

So there are two different backward passes:

### Backward across turns

```text
step_scores -> discounted step_returns
```

This propagates terminal outcome to earlier turns.

### Backward across tokens inside one response

```text
response_level_reward -> last token of response -> token-level returns
```

This propagates the turn reward across tokens of that same response.

These are different levels.

## 17. Reward Normalization in This Exact YAML

This YAML says:

```yaml
reward_normalization:
  grouping: traj_group_id
  method: mean
```

This resolves to:

- `norm_mean_type = group`
- `norm_std_type = None`

So the normalized reward is:

```text
normalized = score - group_mean
```

And with `grouping: traj_group_id`, the group is the set of step samples from one trajectory group.

In this example:

- `rollout_batch_size = 1`
- `group_size = 1`

So in practice, that group is basically the single trajectory's own turns.

## 18. Critical Consequence for This YAML

Now combine these facts:

1. reward source is terminal 0/1
2. `step_reward_gamma = 1.0`
3. terminal success gives step returns like `[1, 1, 1, 1]`
4. `method: mean` subtracts the trajectory mean

Then:

### Success trajectory

```text
raw step returns     = [1, 1, 1, 1]
group mean           = 1
normalized rewards   = [0, 0, 0, 0]
```

### Failure trajectory

```text
raw step returns     = [0, 0, 0, 0]
group mean           = 0
normalized rewards   = [0, 0, 0, 0]
```

So in this exact config, if reward is only terminal binary reward, the response-level training signal can collapse to zero.

That is a very important result.

## 19. So Does This YAML Actually Learn?

In the strict code path of this example, the answer is:

```text
it may produce effectively zero policy gradient signal
```

Why:

- response-level rewards become all zero after mean-centering
- token-level rewards become all zero
- advantages become all zero
- actor loss becomes zero

This is not a vague statement.
It follows directly from the exact combination of:

- sparse terminal reward
- `step_reward_gamma: 1.0`
- `grouping: traj_group_id`
- `method: mean`
- single-trajectory grouping

## 20. Token-Level Reward in This Example

After response-level reward is computed for a step sample:

1. it is placed on the last token of that response
2. all earlier response tokens initially get zero
3. REINFORCE return is computed backward over the response tokens

Sketch:

```text
response tokens:      [t1, t2, t3, t4]
reward placement:     [ 0,  0,  0,  R]
token returns:        [ R,  R,  R,  R]   if gamma = 1
```

Again, this is **inside one response sample**.

It is not across turns anymore at that stage.

## 21. What Loss Is Used Here?

For this YAML:

- `adv_estimator: step_reinforce`
- `actor_train.pg_variant` is not set
- the actor worker defaults to `pg_variant = "vanilla"`

So the policy loss is the vanilla REINFORCE-style loss:

```text
loss_token = - log_prob(token) * advantage(token)
```

aggregated over response tokens.

There is no PPO clipping unless `pg_variant: ppo` is explicitly set.

## 22. Is There KL Loss Here?

Effectively no.

Why:

- `init_kl_coef: 0.0`
- `use_kl_loss` is not enabled
- `enable_reference` is therefore not turned on by config logic
- later the pipeline falls back to `ref_log_probs = old_log_probs`

So KL is effectively zero in this setup.

Also:

- `entropy_loss_coef: 0`

So the total training objective is basically just the vanilla PG term.

And if the advantages are zero, the total useful loss is zero.

## 23. Final Mental Model

Use this model for this example:

```text
task trajectory
  -> multiple assistant turns
  -> each turn stored as one training sample
  -> env reward is mostly terminal (pass/fail after tests)
  -> that terminal outcome is propagated backward across turns
  -> each turn gets one scalar response reward
  -> that scalar is propagated backward across tokens of that same response
  -> actor uses vanilla REINFORCE loss on those token advantages
```

But for this exact YAML:

```text
terminal 0/1 reward
+ gamma = 1.0 across turns
+ mean-centering within the same trajectory
= can collapse all response rewards to zero
```

## 24. Short Answers

### Is this multi-turn?

Yes. Up to 60 assistant turns per trajectory.

### Is the reward chunk-level?

No.

The main unit is one assistant response turn, not one tool-call chunk.

### Is it purely terminal reward?

The reward source in this env is basically terminal pass/fail.

But ROLL converts that sparse reward into per-turn discounted returns afterward.

### Does each tool call get its own reversed discounted reward?

No.

Each **assistant turn** gets one reversed discounted step return.
A turn may contain multiple tool calls.

### What is `anti_call_llm`?

It is the model-service bridge where:

- input = previous model response
- output = next model request

In local mode it is implemented through a file-based request/response protocol.

## 25. Is It HTTP or Filesystem?

The exact answer is:

- `IFLOW -> local model-service` uses **HTTP**
- `local model-service <-> ROLL` uses the **filesystem log protocol**

So it is not "only filesystem" and it is not "only HTTP".
It is a two-hop bridge.

### Exact sketch

```text
inside sandbox
-------------
iflow
  -> POST /v1/chat/completions
  -> http://localhost:8080/v1/chat/completions

local model-service
  -> append request line to LLMService.log
  -> wait for matching response line in LLMService.log

outside sandbox control loop
----------------------------
ROLL
  -> anti_call_llm(index, last_response)
  -> reads request line from file
  -> runs policy model
  -> writes response line to file
```

So the direct answer to your question is:

```text
between IFLOW and model-service: HTTP
between model-service and ROLL in local mode: filesystem
```

## 26. The `anti_call_llm` Protocol, Precisely

There are really two protocol layers.

### Layer A: HTTP protocol used by IFLOW

IFLOW behaves like a normal OpenAI-compatible client.

It sends an HTTP request like:

```text
POST /v1/chat/completions
Content-Type: application/json
```

with body shaped like:

```json
{
  "model": "...",
  "messages": [...],
  "tools": [...]
}
```

The local model-service receives that HTTP request.

### Layer B: file protocol used by model-service and ROLL

The local model-service does **not** answer the HTTP request directly by calling a remote LLM.
Instead it:

1. assigns a request index
2. writes the request into the log file
3. waits for ROLL to write the matching response
4. returns that response back to IFLOW over HTTP

So the file protocol is the internal bridge that powers the HTTP endpoint.

### File markers

```text
LLM_REQUEST_START
LLM_REQUEST_END
LLM_RESPONSE_START
LLM_RESPONSE_END
SESSION_END
```

### Request frame

```text
LLM_REQUEST_START{request_json}LLM_REQUEST_END{meta_json}
```

### Response frame

```text
LLM_RESPONSE_START{response_json}LLM_RESPONSE_END{meta_json}
```

### Session-end frame

```text
SESSION_END
```

### Metadata

The metadata includes at least:

```json
{
  "timestamp": 1740000000000,
  "index": 3
}
```

### Index semantics

`anti_call_llm()` is intentionally inverted:

```text
anti_call_llm(0, None)
  -> wait for first request
  -> return request #1

anti_call_llm(1, response_payload_for_request_1)
  -> write response #1
  -> wait for request #2
  -> return request #2

anti_call_llm(2, response_payload_for_request_2)
  -> write response #2
  -> wait for request #3
  -> return request #3
```

So the meaning is:

- input = previous response
- output = next request

That is why the name is `anti_call_llm`.

## 27. Strict Call Chain: `reset()` to `loss_func()`

This section follows the exact control path for this example.

### Phase A: environment reset and first request

1. `AgentNativeStepEnvManager.run_rollout_loop()` starts a new trajectory.
2. It calls `AgentNativeStepEnvManager.reset()`.
3. That calls `self.env.reset(seed=seed)`.
4. Here `self.env` is `RockTBNativeEnv`, so `RockTBNativeEnv.reset()` runs.
5. `RockTBNativeEnv.reset()` loads one dataset item and extracts:
   - `prompt`
   - `sandbox_image`
   - `task_name`
6. `RockTBNativeEnv.reset()` calls `start_sandbox()`.
7. `start_sandbox()` constructs `SandboxManagerV2(...)`.
8. `SandboxManagerV2` creates the ROCK sandbox and installs the configured agent.
9. The configured agent type is `"default"`, which maps to `RockAgent`.
10. `RockTBNativeEnv.reset()` then calls `reset_agent_status(prompt=self.prompt)`.
11. `reset_agent_status()` calls `sandbox_manager.start_agent(prompt=prompt)`.
12. `SandboxManagerV2.start_agent()` calls `AgentManager.start_agent(prompt)`.
13. `AgentManager.start_agent()` calls `self.agent.run(prompt=prompt)`.
14. `RockAgent.run()` builds the actual shell command from `run_cmd`.
15. In this YAML that command becomes:

```text
iflow -p <<PROMPT>> --yolo
```

16. `RockAgent._agent_run()` starts the agent process in `nohup` mode.
17. `RockAgent._agent_run()` also starts model-service process watching with `/v1/agent/watch`.
18. Now the IFLOW process runs inside the sandbox.
19. IFLOW tries to call the local OpenAI-compatible endpoint:

```text
POST http://localhost:8080/v1/chat/completions
```

20. The local model-service endpoint receives that request.
21. It assigns request index `1`.
22. It writes one `LLM_REQUEST_START...LLM_REQUEST_END...` line to the log file.
23. It then waits for response index `1`.
24. Back in `reset_agent_status()`, ROLL calls:

```text
sandbox_manager.fetch_agent_request(index=0)
```

25. `fetch_agent_request()` calls `AgentManager.anti_call_llm(index=0, response_payload=None)`.
26. That calls ROCK `ModelClient.anti_call_llm(0, None)`.
27. `ModelClient.anti_call_llm(0, None)` waits for the first request to exist in the log.
28. It then returns request `#1`.
29. `reset_agent_status()` receives that request payload.
30. `sandbox_manager.get_messages_and_tools()` parses the JSON and extracts:
   - `messages`
   - `tools`
31. Those become the first observation returned to the env manager.
32. `AgentNativeStepEnvManager.reset()` stores that observation in `rollout_cache.history`.

### Phase B: one rollout turn

1. `AgentNativeStepEnvManager.run_rollout_loop()` calls `make_decision(rollout_cache)`.
2. `make_decision()` calls `format_messages(rollout_cache)`.
3. `format_messages()` takes the current observation messages and tool schema.
4. It applies the tokenizer chat template and creates:
   - `input_ids`
   - `attention_mask`
   - `position_ids`
5. `make_decision()` then calls:

```text
llm_proxy.generate(messages=input_messages, lm_input=lm_input, generation_config=...)
```

6. `PolicyProxy.generate()` sends the request to the actor-infer model service in ROLL.
7. The policy model returns one assistant response.
8. `make_decision()` stores:
   - `response_ids`
   - optional `infer_logprobs`
   - assistant text appended to `messages`

### Phase C: feeding that response back to IFLOW

1. `AgentNativeStepEnvManager.step(llm_output)` decodes the response text.
2. It calls `RockTBNativeEnv.step(action=response_text)`.
3. `RockTBNativeEnv.step()` calls:

```text
sandbox_manager.format_response_payload(response=action)
```

4. `format_response_payload()` parses textual tool-call markup such as:
   - `<tool_call>...</tool_call>`
   - `<function=...>...</function>`
5. It converts that text into an OpenAI-style response payload.
6. `RockTBNativeEnv.step()` then calls:

```text
sandbox_manager.fetch_agent_request(
    index=self.current_session_step,
    response_payload=response_payload
)
```

7. `fetch_agent_request()` again reaches `ModelClient.anti_call_llm(...)`.
8. If `current_session_step == 1`, then this is:

```text
anti_call_llm(1, response_payload_for_request_1)
```

9. `ModelClient.push_response()` writes:

```text
LLM_RESPONSE_START{response_json}LLM_RESPONSE_END{meta_json}
```

for index `1`.
10. The local model-service sees that response line while it is polling.
11. The model-service returns that JSON payload as the HTTP response to IFLOW.
12. IFLOW receives the assistant message over HTTP.
13. IFLOW may execute one or more tools inside the sandbox.
14. After tool execution, IFLOW issues the next HTTP `/v1/chat/completions` call.
15. The model-service assigns request index `2`.
16. It writes request `#2` into the log file.
17. `anti_call_llm(1, response_1)` returns request `#2` back to ROLL.
18. `RockTBNativeEnv.step()` parses request `#2` into the next `messages` and `tools`.
19. That next observation is appended into `rollout_cache.history`.

This is why one trajectory is multi-turn:

```text
request_1 -> response_1 -> request_2 -> response_2 -> request_3 -> ...
```

### Phase D: termination and raw reward

1. At some point IFLOW exits, or max steps is reached.
2. If IFLOW exits naturally, the watch-agent endpoint writes `SESSION_END`.
3. `RockTBNativeEnv.step()` receives `next_request_payload == "SESSION_END"`.
4. It calls `check_terminated(next_request_payload="SESSION_END")`.
5. `check_terminated()` then calls `calculate_reward()`.
6. `calculate_reward()` calls:

```text
sandbox_manager.run_tests(...)
```

7. `run_tests()` uploads the task tests and `run-tests.sh`.
8. It runs the tests inside the sandbox.
9. It parses the output into terminal success/failure.
10. That becomes the env reward for the final step.

So the raw env-side step rewards look like:

```text
success case: [0, 0, 0, 1]
failure case: [0, 0, 0, 0]
```

### Phase E: turning the finished trajectory into training samples

1. Once `rollout_cache.terminated` is true, the env manager calls `formulate_rollouts()`.
2. `formulate_rollouts()` iterates over every assistant turn in the trajectory.
3. For each turn it creates exactly one training sample.
4. For that sample it stores:
   - prompt tokens
   - response tokens
   - `step_scores = history["reward"]`
   - `step = turn_index`
5. It also writes `score_tensor[0][-1] = history["reward"]`.

Important:

- this is one sample per assistant response turn
- it is not one sample per tool call

### Phase F: discounted step returns across turns

1. The training pipeline receives the rollout batch.
2. `AgenticPipeline` calls:

```text
compute_discounted_returns(batch, adv_estimator="step_reinforce", gamma=1.0)
```

3. That groups samples by `traj_id`.
4. Inside each trajectory it sorts by `step`.
5. It reads `non_tensor_batch["step_scores"]`.
6. It computes:

```text
G_t = r_t + gamma * G_{t+1}
```

backward over turns.
7. With terminal-only success `[0, 0, 0, 1]` and `gamma = 1.0`, this becomes:

```text
step_rewards = [1, 1, 1, 1]
```

So yes:

- the reward source is terminal-like
- but the training batch becomes per-turn returns afterward

That means the terminal outcome is propagated backward across assistant turns.

### Phase G: response-level reward normalization

1. `AgenticPipeline` calls `compute_response_level_rewards(...)`.
2. For `adv_estimator == "step_reinforce"`, it uses `batch.batch["step_rewards"]`.
3. It applies `agentic_reward_norm(...)`.
4. In this YAML:

```yaml
reward_normalization:
  grouping: traj_group_id
  method: mean
```

5. So each turn reward is mean-centered within its trajectory group.
6. With one successful trajectory:

```text
raw step_rewards        = [1, 1, 1, 1]
mean-centered rewards   = [0, 0, 0, 0]
```

7. With one failed trajectory:

```text
raw step_rewards        = [0, 0, 0, 0]
mean-centered rewards   = [0, 0, 0, 0]
```

This is the exact reason the learning signal can collapse in this config.

### Phase H: token-level reward inside each response

1. `AgenticPipeline` calls `compute_token_reward(...)`.
2. `compute_token_reward()` first calls `expand_to_token_level(...)`.
3. `expand_to_token_level()` places the scalar `response_level_reward` on the last token of the sample.
4. Then later `agentic_compute_advantage(...)` with `step_reinforce` calls `compute_reinforce_return(...)`.
5. `compute_reinforce_return()` runs backward over tokens in that one response.

So the structure is:

```text
across turns:
  terminal raw reward -> per-turn discounted returns

inside one turn:
  one scalar response reward -> placed on last token -> backward token returns
```

That is why the correct unit is:

- across turns: assistant turn
- across tokens: tokens inside one assistant response

Not tool-call chunks.

### Phase I: final actor loss

1. After token rewards are prepared, `AgenticPipeline` calls `agentic_compute_advantage(...)`.
2. That writes `advantages` into the batch.
3. Then `AgenticPipeline` starts actor training with:

```text
actor_train.train_step(batch)
```

4. `BaseActorWorker.train_step()` iterates mini-batches and calls:

```text
self.strategy.train_step(batch=backward_batch, loss_func=self.loss_func)
```

5. In this example the actual override is `agentic_actor_pg_worker.ActorWorker.loss_func(...)`.
6. That function reads:
   - `response_mask`
   - `ref_log_probs`
   - `advantages`
7. It computes fresh `log_probs` from the current model output.
8. It computes `old_log_probs`.
9. Because `pg_variant` defaults to `"vanilla"`, it uses vanilla policy gradient:

```text
pg_loss = -log_probs * advantages
```

through `_compute_vanilla_pg_loss(...)`, then aggregates over response tokens.
10. KL is effectively inactive here because reference is disabled and KL coefficient is zero.
11. Entropy is also inactive because `entropy_loss_coef: 0`.
12. So in practice:

```text
total_loss ~= vanilla policy-gradient loss
```

13. If advantages are all zero after the reward path above, then this loss becomes effectively zero too.

## 28. Final Correction to Your Interpretation

Your revised statement should be:

```text
The raw env reward is basically terminal test pass/fail.
After the rollout ends, ROLL propagates that outcome backward across assistant turns with compute_discounted_returns().
Then each assistant-turn sample gets one scalar response-level reward.
Then that scalar is propagated backward across the tokens of that one response.
This is not chunked by tool call.
```

So:

- yes, it is not "only terminal" once training preprocessing starts
- yes, the terminal outcome is propagated backward
- no, the unit is not tool-call chunks
- the main training unit is the assistant turn
