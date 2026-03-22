#!/bin/bash
# install_up_roll.sh — Reproducible ROLL container setup
# Creates Docker container with all ROLL dependencies for agentic SWE pipeline
set -euo pipefail

IMAGE="roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-25.11-py3-torch2100-mcore0160dev-vllm016dev"
CONTAINER="roll_swe_runner"
ALE_ROOT="/home/ubuntu/ALE-latest"
ROLL_DIR="${ALE_ROOT}/ROLL-personal"

echo "=== Phase 1: Docker Container Setup ==="

# Remove existing container if any
docker rm -f ${CONTAINER} 2>/dev/null || true

# Create container
# --pid=host + SYS_PTRACE + seccomp=unconfined: required for CUDA IPC (pidfd_getfd)
# used by Mode B (partial GPU, simultaneous train+infer) colocated model update
docker run -dit \
    --cap-add SYS_ADMIN \
    --cap-add SYS_PTRACE \
    --gpus all \
    --name ${CONTAINER} \
    --ipc=host \
    --pid=host \
    --security-opt seccomp=unconfined \
    --security-opt label=disable \
    --net=host \
    -v /home/ubuntu/:/home/ubuntu/ \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -w ${ROLL_DIR} \
    ${IMAGE} \
    /bin/bash

echo "Container created: ${CONTAINER}"

# Set NCCL env vars (critical: avoids segfaults from AWS OFI NCCL plugin)
docker exec ${CONTAINER} bash -c "
    echo 'export NCCL_NET_PLUGIN=\"\"' >> /root/.bashrc
    echo 'export NCCL_TUNER_PLUGIN=\"\"' >> /root/.bashrc
    echo 'export NCCL_NET=Socket' >> /root/.bashrc
"
echo "NCCL env vars configured"

# Install ROLL + dependencies
docker exec ${CONTAINER} bash -c "
    export NCCL_NET_PLUGIN='' && export NCCL_TUNER_PLUGIN='' && export NCCL_NET=Socket && \
    cd ${ROLL_DIR} && \
    pip install -e . && \
    pip install -e ./mcore_adapter && \
    pip install rl-rock && \
    pip install -r requirements_common.txt
"
echo "ROLL + dependencies installed"

# Create symlink for dataset path (config references /ROLL/data/...)
docker exec ${CONTAINER} ln -sf ${ROLL_DIR} /ROLL
echo "Symlink /ROLL created"

# Clone terminal-bench-datasets (test harnesses for SWE-bench reward computation)
docker exec ${CONTAINER} bash -c "
    if [ ! -d /terminal-bench-datasets ]; then
        git clone https://github.com/laude-institute/terminal-bench-datasets.git /terminal-bench-datasets
    else
        echo 'terminal-bench-datasets already exists'
    fi
"
echo "terminal-bench-datasets ready"

# Stop stale Ray processes
docker exec ${CONTAINER} bash -c "ray stop --force 2>/dev/null || true"

# Verify critical imports
echo "=== Verifying imports ==="
docker exec ${CONTAINER} python -c "
import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
import vllm; print(f'vLLM: {vllm.__version__}')
import megatron; print('megatron-core OK')
import transformers; print(f'transformers: {transformers.__version__}')
import roll; print('ROLL OK')
import mcore_adapter; print('mcore_adapter OK')
from fla.layers import GatedDeltaNet; print('flash-linear-attention (GDN) OK')
import ray; print(f'Ray: {ray.__version__}')
from rock.sdk.sandbox.client import Sandbox; print('rl-rock SDK OK')
"

echo ""
echo "=== ROLL container setup complete ==="
echo "To run the Qwen3-4B SWE pipeline:"
echo "  docker exec -it ${CONTAINER} bash -c \""
echo "    export NCCL_NET_PLUGIN='' && export NCCL_TUNER_PLUGIN='' && export NCCL_NET=Socket && \\"
echo "    cd ${ROLL_DIR} && \\"
echo "    export PYTHONPATH='${ROLL_DIR}:\\\$PYTHONPATH' && \\"
echo "    python examples/start_agentic_pipeline.py --config_path agentic_demo --config_name agent_val_rock_swe"
echo "  \""
