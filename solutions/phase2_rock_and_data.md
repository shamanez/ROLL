# Phase 2: ROCK + Data Environment Prep - Solution

## What Was Done
1. Verified ROCK Admin at localhost:8080
2. Created /ROLL symlink for dataset paths
3. Cloned terminal-bench-datasets for SWE-bench test harnesses
4. **Fixed two ROCK bugs** (see below)

## ROCK Bug Fix 1: Rocklet Python symlink resolution (runtime_env.py)

**File:** `ROCK-personal/rock/deployments/runtime_env.py`

**Problem:** ROCK's `LocalRuntimeEnv` mounts the uv-managed Python at its real path (`cpython-3.11.15-linux-x86_64-gnu`) but the `.venv/bin/python` symlink inside the container points to the SYMLINK path (`cpython-3.11-linux-x86_64-gnu`). This symlink doesn't exist in the container, so the rocklet process fails to start with "bad interpreter".

**Root cause:** uv creates a symlink `cpython-3.11-linux-x86_64-gnu` → `cpython-3.11.15-linux-x86_64-gnu` on the host. Docker volume mounts only mount real directories, not symlinks.

**Fix:** Added code to `LocalRuntimeEnv.get_volume_mounts()` that resolves the `.venv/bin/python` symlink chain and adds the symlink parent directory as an additional volume mount pointing to the real directory. This way both the real path AND the symlink path resolve inside the container.

## ROCK Bug Fix 2: PhaseStatus JSON serialization (sandbox_manager.py)

**File:** `ROCK-personal/rock/sandbox/sandbox_manager.py`

**Problem:** `get_status()` stores `SandboxInfo` in Redis, but the `phases` field contains `PhaseStatus` (Pydantic model) and `state` field contains `State` (enum), which aren't JSON-serializable.

**Fix:** Added serialization of `PhaseStatus` objects (via `.to_dict()`) and enum values (via `.value`) before storing in Redis.
