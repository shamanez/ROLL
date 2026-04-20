# Phase 1: Docker Container Setup - Solution

## What Was Done

Container `roll_swe_runner` created using the pre-built ROLL image with all CUDA dependencies.

## Commands
```bash
# IMPORTANT: --pid=host, SYS_PTRACE, and seccomp=unconfined are REQUIRED for Mode B
# (partial GPU, simultaneous train+infer). Without these, CUDA IPC fails with:
# RuntimeError: pidfd_getfd: Operation not permitted
docker run -dit \
    --cap-add SYS_ADMIN --cap-add SYS_PTRACE \
    --gpus all --name roll_swe_runner \
    --ipc=host --pid=host \
    --security-opt seccomp=unconfined --security-opt label=disable \
    --net=host \
    -v /home/ubuntu/:/home/ubuntu/ -v /var/run/docker.sock:/var/run/docker.sock \
    -w /home/ubuntu/ALE-latest/ROLL-personal \
    roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-25.11-py3-torch2100-mcore0160dev-vllm016dev /bin/bash

# NCCL env vars (critical - prevents segfaults from AWS OFI NCCL plugin)
docker exec roll_swe_runner bash -c "echo 'export NCCL_NET_PLUGIN=\"\"' >> /root/.bashrc; echo 'export NCCL_TUNER_PLUGIN=\"\"' >> /root/.bashrc; echo 'export NCCL_NET=Socket' >> /root/.bashrc"

# Install ROLL + dependencies
docker exec roll_swe_runner bash -c "cd /home/ubuntu/ALE-latest/ROLL-personal && pip install -e . && pip install -e ./mcore_adapter && pip install rl-rock && pip install -r requirements_common.txt"
```

## Verification
All imports succeed: PyTorch 2.10.0+cu130, 8 GPUs, vLLM 0.16.0rc2, megatron-core, transformers 5.2.0, flash-linear-attention GDN, Ray 2.48.0, rl-rock SDK.
