# ROCK + ROLL Logging Investigation: All Ways to View Agent-Sandbox Interaction Logs

This document catalogs every logging mechanism found in the ROCK and ROLL codebases that can capture the textual flow between the model and sandbox (prompts, tool calls, tool outputs, agent responses, rewards).

---

## 1. Rocklet HTTP Middleware Logging (Inside Container)

**File:** `/home/ubuntu/ALE-latest/ROCK-personal/rock/rocklet/server.py` (lines 28-70)

### What It Logs
The `log_requests_and_responses` middleware intercepts every HTTP request to the Rocklet server (except `/SandboxFusion` paths). It logs:
- **Request:** method, URL, headers, full JSON request body (including `sandbox_id`)
- **Response:** status code, processing time in milliseconds

This captures every command execution, session creation, file upload, env step/reset -- essentially all sandbox operations.

### Where Logs Go
The middleware creates a separate logger via:
```python
req_logger = init_logger("rocklet.accessLog", "access.log")
```
- **Without `ROCK_LOGGING_PATH`:** Logs go to stdout (container stdout, visible via `docker logs`)
- **With `ROCK_LOGGING_PATH` set:** Logs write to `<ROCK_LOGGING_PATH>/access.log` as a file

### How to Enable File Logging
Set the `ROCK_LOGGING_PATH` environment variable before starting the rocklet:
```bash
# Inside the container or via Docker env:
export ROCK_LOGGING_PATH=/data/logs
rocklet --port 8000
```

The `init_file_handler` in `rock/logger.py` (lines 68-79) creates a `FileHandler` writing to `<ROCK_LOGGING_PATH>/<file_name>`. For the access log specifically, the file will be `<ROCK_LOGGING_PATH>/access.log`.

### Log Format
```
2026-03-19T15:02:10.123+08:00 INFO:server.py:56 [rocklet.accessLog] [sandbox-abc123] [trace-xyz] -- {
  "access_type": "request",
  "method": "POST",
  "url": "http://0.0.0.0:8000/run_in_session",
  "headers": {"content-type": "application/json", ...},
  "request_content": {"session": "agent", "command": "ls -la /app", "check": "silent"}
}
```
Followed by:
```
... "access_type": "response", "status_code": 200, "process_time": "45.23ms" ...
```

**Limitation:** The response middleware only logs status code and timing, NOT the response body. To see response bodies, you need the model-service trajectory logging (mechanism #6 below).

### How to Extract From Running Containers
If `ROCK_LOGGING_PATH` was not set, access the container's stdout:
```bash
# Via ROCK SDK:
sandbox.arun(cmd="cat /data/logs/access.log")
# Or via docker:
docker logs <container_id> 2>&1 | grep "access_type"
```

---

## 2. IFlow CLI Session Logs (Inside Container)

**File:** `/home/ubuntu/ALE-latest/ROCK-personal/rock/sdk/sandbox/agent/iflow_cli.py`

### What It Logs
The iflow CLI agent's stdout/stderr is redirected to a log file:
```python
iflow_cmd = f'iflow -r "{session_id}" -p {shlex.quote(prompt)} --yolo > {self.config.iflow_log_file} 2>&1'
```
Default path: `~/.iflow/session_info.log` (configurable via `IFlowCliConfig.iflow_log_file`)

This file contains the full iflow CLI output -- all agent reasoning, tool call decisions, tool execution results, and session metadata including an `<Execution Info>` block with session-id JSON.

### Where Logs Are Stored
Inside the container at: `~/.iflow/session_info.log` (typically `/root/.iflow/session_info.log`)

### How to Extract From Running Containers
```python
# Via ROCK SDK:
result = await sandbox.arun(cmd="cat ~/.iflow/session_info.log")
print(result.output)

# Via sandbox_manager_v2 (ROLL side):
response = sandbox_manager.run_in_session("cat ~/.iflow/session_info.log", "agent")
```

### What the Output Looks Like
Contains the full iflow CLI output including:
- System prompt
- User messages
- Tool call selections
- Tool execution outputs
- Session metadata in `<Execution Info>{ "session-id": "..." }</Execution Info>` format

**Note:** This file is used by `_get_session_id_from_sandbox()` (lines 180-222) to extract session IDs for checkpoint/resume. The code reads the last 1000 lines and parses the `<Execution Info>` block.

**Important:** In the ROLL pipeline's native mode (SWE pipeline), the `RockAgent._create_agent_run_cmd` is used instead of `IFlowCli._create_agent_run_cmd`, so this specific log file may not be created. The native mode uses the model-service approach (mechanism #6) rather than direct iflow CLI invocation.

---

## 3. ROCK Environment Variables for Logging

**File:** `/home/ubuntu/ALE-latest/ROCK-personal/rock/env_vars.py`

### Three Key Variables

| Variable | Default | Purpose |
|---|---|---|
| `ROCK_LOGGING_PATH` | `None` (unset = stdout) | Directory for log files. When set, ALL ROCK loggers write to files in this directory |
| `ROCK_LOGGING_FILE_NAME` | `"rocklet.log"` | Default log filename for general ROCK loggers |
| `ROCK_LOGGING_LEVEL` | `"INFO"` | Log level (DEBUG, INFO, WARNING, ERROR) |

### How the Logging System Works (rock/logger.py)

The `init_logger(name, file_name)` function:
1. If `ROCK_LOGGING_PATH` is set AND `file_name` is provided: writes to `<ROCK_LOGGING_PATH>/<file_name>`
2. If `ROCK_LOGGING_PATH` is not set: writes to stdout with colored output
3. The file handler uses `mode="w+"` (overwrite), NOT append mode
4. Format includes: timestamp, level, file:line, logger name, sandbox_id (from context var), trace_id

### How to Enable Full Debug Logging to Files
```bash
export ROCK_LOGGING_PATH=/data/logs
export ROCK_LOGGING_FILE_NAME=rocklet.log
export ROCK_LOGGING_LEVEL=DEBUG
```

This creates two log files when rocklet runs:
- `/data/logs/rocklet.log` -- general rocklet logs
- `/data/logs/access.log` -- HTTP access logs (from middleware)

### Additional Logging Variables

| Variable | Default | Purpose |
|---|---|---|
| `ROCK_MODEL_SERVICE_DATA_DIR` | `"/data/logs"` | Directory for model service logs and trajectory files |
| `ROCK_MODEL_SERVICE_TRAJ_APPEND_MODE` | `false` | If true, append to trajectory file; if false, overwrite |
| `ROCK_TIME_ZONE` | `"Asia/Shanghai"` | Timezone for log timestamps |

---

## 4. SandboxManager / SandboxActor Logs

### SandboxActor (rock/sandbox/sandbox_actor.py)
Logs sandbox lifecycle events:
- Container start/stop
- Background script execution
- Docker commit operations
- Health check failures

Logger: `init_logger(__name__)` -- goes to stdout or file per ROCK_LOGGING_PATH.

### BaseSandboxManager (rock/sandbox/base_manager.py)
Logs:
- APScheduler start/stop for metrics collection
- Sandbox monitoring intervals
- Metrics collection for active sandboxes
- Timeout warnings

### SandboxProxyService (rock/sandbox/service/sandbox_proxy_service.py)
Logs:
- Port forwarding operations
- WebSocket proxy connections
- Redis-based sandbox status queries
- Sandbox listing/filtering operations

All these use the standard ROCK logger pattern: `logger = init_logger(__name__)`.

---

## 5. SDK Client-Side Logging

**File:** `/home/ubuntu/ALE-latest/ROCK-personal/rock/sdk/sandbox/client.py`

### What It Logs
The `Sandbox` client uses Python's standard `logging` module (NOT `rock.logger`):
```python
logger = logging.getLogger(__name__)
```

It logs at `DEBUG` level:
- Start sandbox responses
- Get status responses
- Execute command responses
- Create/close session responses
- Upload responses
- Run-in-session responses

And at `INFO`/`WARNING` level:
- Sandbox start with image info
- Nohup process PIDs
- File read operations
- Failed sandbox operations with retry info

### How to Enable
Since it uses standard Python logging, configure it in your application:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
# Or specifically for the SDK:
logging.getLogger("rock.sdk.sandbox.client").setLevel(logging.DEBUG)
```

### What It Does NOT Log
The SDK client logs request metadata (URLs, sandbox IDs) but does NOT log the full request/response bodies of commands. For that, you need the Rocklet middleware (mechanism #1).

---

## 6. ROCK Model Service Trajectory Logging (Inside Container)

**Files:**
- `/home/ubuntu/ALE-latest/ROCK-personal/rock/sdk/model/server/utils.py`
- `/home/ubuntu/ALE-latest/ROCK-personal/rock/sdk/model/server/config.py`
- `/home/ubuntu/ALE-latest/ROCK-personal/rock/sdk/model/server/api/local.py`
- `/home/ubuntu/ALE-latest/ROCK-personal/rock/sdk/model/server/api/proxy.py`

### What It Logs
**THIS IS THE MOST IMPORTANT MECHANISM for seeing actual agent-model interactions.**

The `@record_traj` decorator on `/v1/chat/completions` captures:
- **Request:** Full OpenAI-compatible chat completion request including all messages (system, user, assistant, tool results), tools definitions
- **Response:** Full model response including generated content, tool calls, finish_reason

Every LLM interaction that goes through the model service is recorded.

### Where Logs Are Stored
Inside the container at: `/data/logs/LLMTraj.jsonl` (configurable via `ROCK_MODEL_SERVICE_DATA_DIR`)

The path is computed in `config.py`:
```python
LOG_DIR = env_vars.ROCK_MODEL_SERVICE_DATA_DIR  # default: "/data/logs"
TRAJ_FILE = LOG_DIR + "/LLMTraj.jsonl"
LOG_FILE = LOG_DIR + "/LLMService.log"
```

### Format
JSONL format, one JSON object per line:
```json
{
  "request": {
    "model": "ROME",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant..."},
      {"role": "user", "content": "Fix the bug in django/core/..."},
      {"role": "assistant", "content": "Let me look at the code...", "tool_calls": [...]},
      {"role": "tool", "content": "file contents here..."}
    ],
    "tools": [...],
    "temperature": 1.0
  },
  "response": {
    "choices": [{
      "index": 0,
      "message": {"role": "assistant", "content": "I see the issue...", "tool_calls": [...]},
      "finish_reason": "tool_calls"
    }]
  }
}
```

### How to Configure
- `ROCK_MODEL_SERVICE_DATA_DIR` (default `/data/logs`): directory for traj file
- `ROCK_MODEL_SERVICE_TRAJ_APPEND_MODE` (default `false`): set to `true` to append instead of overwrite

### How to Extract From Running Containers
```python
# During a run, extract from the sandbox:
result = await sandbox.arun(cmd="cat /data/logs/LLMTraj.jsonl")

# Or via sandbox_manager_v2 (ROLL side):
response = sandbox_manager.run_in_session("cat /data/logs/LLMTraj.jsonl", "agent")
```

### Model Service General Log
Additionally, `/data/logs/LLMService.log` contains the model service operational logs. The model service startup command explicitly sets logging:
```python
bash_start_cmd = (
    f"export ROCK_LOGGING_PATH={self.config.logging_path} && "
    f"export ROCK_LOGGING_FILE_NAME={self.config.logging_file_name} && "
    ...
)
```

---

## 7. ROLL-Side Trajectory Logging and Dumping

### 7a. `dump_rollout_trajectories` (Persistent Storage)

**Files:**
- `/home/ubuntu/ALE-latest/ROLL-personal/roll/pipeline/agentic/utils.py` (lines 298-334)
- `/home/ubuntu/ALE-latest/ROLL-personal/roll/pipeline/rlvr/utils.py` (lines 37-52)

### What It Logs
Dumps trajectory data to disk as JSONL files. For the agentic pipeline, each trajectory includes:
- `trajectory_data`: Full trajectory metadata (trajectory_id, env_info, timing_info, reward_info, failure_info, metrics, last_observation)
- `messages`: Complete conversation history (all messages exchanged)
- `tools`: Tool definitions used
- `exp_name`: Experiment name
- `global_step`: Training step number

### Where Logs Are Stored
Configured via `rollout_dump_dir` in the YAML config:
```yaml
rollout_dump_dir: ./output/rollout_dump
```
Files are written as: `<rollout_dump_dir>/rollout_dump_data.step_<N>.jsonl`

The current configs all set this to `./output/rollout_dump` (relative to ROLL-personal working dir).

### How to Enable
Set `rollout_dump_dir` to a valid absolute path in your YAML config. It must start with `/` for the `json_checker` to return true. The current default `./output/rollout_dump` is a relative path that will NOT trigger file writing because `json_checker` checks `path.startswith("/")`.

**To fix this and enable dumping:**
```yaml
rollout_dump_dir: /home/ubuntu/ALE-latest/ROLL-personal/output/rollout_dump
```

### When It Fires
- After each training step's `get_batch` (line 289 of agentic_pipeline.py)
- After each validation step (line 595 of agentic_pipeline.py)
- After each rollout-only step (line 102 of agentic_rollout_pipeline.py)

### File Format
JSONL with columns defined by `COLUMMNS_CONFIG`:
```json
{
  "trajectory_data": "{\"trajectory_id\":\"...\",\"env_info\":{\"task_name\":\"django__django-12345\",...},\"reward_info\":{\"episode_reward\":1.0,...},\"failure_info\":{...},\"last_observation\":[...]}",
  "messages": "[[{\"role\":\"system\",\"content\":\"...\"},{\"role\":\"user\",\"content\":\"...\"},...]]]",
  "tools": "[{\"type\":\"function\",\"function\":{\"name\":\"Shell\",...}}]",
  "exp_name": "agentic_rollout_swe",
  "global_step": 0
}
```

**CRITICAL NOTE:** The `trajectory_data` is stored as a JSON string inside the JSONL (double-encoded). You need to `json.loads()` the value to access the nested structure.

### 7b. Pipeline Logging (Console/File)

**File:** `/home/ubuntu/ALE-latest/ROLL-personal/roll/pipeline/agentic/agentic_pipeline.py` (lines 520-563)

Every `logging_steps` steps, the pipeline logs to console:
- Decoded prompts and responses for up to 10 trajectory groups
- Episode scores and step scores
- Full metrics dictionary

This goes to the ROLL logger output (typically stdout of the Ray driver).

### 7c. Existing Log Files Found on This System

**ROLL output logs:** `/home/ubuntu/ALE-latest/ROLL-personal/output/logs/`
- `EnvironmentWorker(train_env-0).log` -- Environment worker initialization
- `EnvironmentWorker(val_env-0).log` -- Validation env worker
- `ActorWorker(actor_train-*.log` -- Training worker logs
- `InferWorker(actor_infer-*.log` -- Inference worker logs
- `RolloutScheduler.log` -- Rollout scheduling
- `log_rank_DRIVER_0_1.log` -- Pipeline driver log (contains the actual prompt/response logs from the pipeline)

**ROLL tensorboard events:** `/home/ubuntu/ALE-latest/ROLL-personal/output/agentic_rollout_swe/`
- Contains TensorBoard event files with scalar metrics (scores, loss, timing) -- NOT textual logs.

**No rollout_dump files exist** because `rollout_dump_dir: ./output/rollout_dump` uses a relative path that fails the `startswith("/")` check.

---

## 8. ROLL-Side Environment Manager Logging

**File:** `/home/ubuntu/ALE-latest/ROLL-personal/roll/pipeline/agentic/env/sandbox/rock_tb_native_env.py`

### What It Logs
The `RockTBNativeEnv` class logs extensively at each stage:
- `[SANDBOX_INIT]` -- Sandbox creation with image name
- `[ENV_RESET]` -- Task ID, name, sandbox IP/ID
- `[ENV_STEP]` -- Step number, response text, reward, success status
- `[REWARD_CALC]` -- Reward calculation for each episode
- `[TEST_SESSION]` -- Test execution and results
- Various failure modes with `[FAILED!]` markers

These logs go to the ROLL logger (typically Ray worker stdout, captured in the EnvironmentWorker log files).

### Where to Find These Logs
In the Ray worker logs:
- `/home/ubuntu/ALE-latest/ROLL-personal/output/logs/EnvironmentWorker(train_env-N).log`
- Or in Ray's log directory: `/tmp/ray/session_latest/logs/`

---

## 9. SandboxManagerV2 Logging (ROLL Side, Detailed Interaction Logs)

**File:** `/home/ubuntu/ALE-latest/ROLL-personal/roll/pipeline/agentic/env/rock/sandbox_manager_v2.py`

### What It Logs
Extremely detailed logs of every sandbox interaction:
- `[SANDBOX_INIT]` / `[SANDBOX_START]` -- Sandbox lifecycle with IP/ID
- `[SESSION_CREATE]` -- Bash session creation
- `[AGENT_INSTALL]` -- IFlow CLI installation
- `[RUN_SESSION]` -- Every command execution with retry status
- `[FORMAT_RESPONSE]` -- Response payload formatting (tool calls detected)
- `[GET_MESSAGES_TOOLS]` -- Message/tool extraction from payloads
- `[RUN_TESTS]` -- Test execution and results
- `[GROUND_TRUTH]` -- Ground truth solution execution

At DEBUG level, it logs the actual command text and output lengths. At INFO level, it logs step completion with rewards.

---

## Summary: Quick Reference for Enabling Full Interaction Logging

### To see ALL text flowing between model and sandbox:

**Option A: Model Service Trajectory File (Recommended)**

This gives you the cleanest view of the actual messages and tool calls:

1. The model service inside the container automatically writes `/data/logs/LLMTraj.jsonl`
2. Set `ROCK_MODEL_SERVICE_TRAJ_APPEND_MODE=true` to keep all interactions (not just the last one)
3. Extract after a run: `sandbox.arun(cmd="cat /data/logs/LLMTraj.jsonl")`

**Option B: ROLL Trajectory Dump (Post-Training)**

1. Change `rollout_dump_dir` in your YAML config to an absolute path:
   ```yaml
   rollout_dump_dir: /home/ubuntu/ALE-latest/ROLL-personal/output/rollout_dump
   ```
2. After a training run, read files at that path:
   ```python
   import json
   with open("/home/ubuntu/ALE-latest/ROLL-personal/output/rollout_dump/rollout_dump_data.step_0.jsonl") as f:
       data = json.loads(f.readline())
       messages = json.loads(data["messages"])
       trajectory = json.loads(data["trajectory_data"])
   ```

**Option C: Rocklet Access Logs (Low-Level HTTP)**

1. Set inside containers: `ROCK_LOGGING_PATH=/data/logs` and `ROCK_LOGGING_LEVEL=DEBUG`
2. Read `/data/logs/access.log` for all HTTP request bodies

**Option D: ROLL Pipeline Console Logs (Real-Time)**

The pipeline already logs decoded prompts/responses every `logging_steps`. Check:
- The Ray driver's stdout
- `/home/ubuntu/ALE-latest/ROLL-personal/output/logs/log_rank_DRIVER_0_1.log`

### Environment Variables Cheat Sheet

| Variable | Where to Set | Effect |
|---|---|---|
| `ROCK_LOGGING_PATH=/data/logs` | Container env | Enables file logging for all ROCK components |
| `ROCK_LOGGING_LEVEL=DEBUG` | Container env | Verbose logging including command details |
| `ROCK_MODEL_SERVICE_DATA_DIR=/data/logs` | Container env (default) | Directory for LLMTraj.jsonl |
| `ROCK_MODEL_SERVICE_TRAJ_APPEND_MODE=true` | Container env | Append mode for trajectory file |
| `rollout_dump_dir: /absolute/path` | ROLL YAML config | Enables trajectory dump to disk |
