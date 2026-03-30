#!/bin/bash
# install_up_roll.sh — Reproducible ROLL container setup
# Creates Docker container with all ROLL dependencies for agentic SWE pipeline
set -euo pipefail

IMAGE="roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-25.11-py3-torch2100-mcore0160dev-vllm016dev"
CONTAINER="roll_rollouts_as"
ALE_ROOT="/home/ubuntu/ALE-new"
ROLL_DIR="${ALE_ROOT}/ROLL"

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

echo "=== Phase 2: NCCL Environment ==="

# Set NCCL env vars (critical: avoids segfaults from AWS OFI NCCL plugin)
docker exec ${CONTAINER} bash -c "
    echo 'export NCCL_NET_PLUGIN=\"\"' >> /root/.bashrc
    echo 'export NCCL_TUNER_PLUGIN=\"\"' >> /root/.bashrc
    echo 'export NCCL_NET=Socket' >> /root/.bashrc
"
echo "NCCL env vars configured"

echo "=== Phase 3: Install ROLL ==="

docker exec ${CONTAINER} bash -c "
    export NCCL_NET_PLUGIN='' && export NCCL_TUNER_PLUGIN='' && export NCCL_NET=Socket && \
    cd ${ROLL_DIR} && \
    pip install -e . && \
    pip install -e ./mcore_adapter && \
    pip install -r requirements_common.txt
"
echo "ROLL installed"

echo "=== Phase 4: Install OpenReward SDK ==="

docker exec ${CONTAINER} bash -c "
    pip install openreward
"
echo "OpenReward SDK installed"

echo "=== Phase 5: Symlinks ==="

# Create symlink for dataset path (config references /ROLL/data/...)
docker exec ${CONTAINER} ln -sf ${ROLL_DIR} /ROLL
echo "Symlink /ROLL created"

echo "=== Phase 6: Stop stale Ray ==="

docker exec ${CONTAINER} bash -c "ray stop --force 2>/dev/null || true"

echo ""
echo "=== Done ==="
echo "Container: ${CONTAINER}"
echo ""
echo "To run training:"
echo ""
echo "  # 1. Create a tmux session (so the run survives SSH disconnects)"
echo "  tmux new -s roll"
echo ""
echo "  # 2. Enter the Docker container interactively"
echo "  docker exec -it ${CONTAINER} bash"
echo ""
echo "  # 3. Set environment variables"
echo "  export NCCL_NET_PLUGIN='' && export NCCL_TUNER_PLUGIN='' && export NCCL_NET=Socket"
echo "  cd ${ROLL_DIR}"
echo "  export PYTHONPATH=\"\$PWD:\$PYTHONPATH\""
echo "  export OPENREWARD_API_KEY=\"your-key-here\""
echo "  export WANDB_API_KEY=\"your-key-here\""
echo ""
echo "  # 4. Launch training"
echo "  python examples/start_agentic_pipeline.py \\"
echo "    --config_path agentic_demo \\"
echo "    --config_name openreward_endless_terminals_IPA_qwen35_2b_v2"
echo ""
echo "  # To detach from tmux: Ctrl+b then d"
echo "  # To reattach later:   tmux attach -t roll"
