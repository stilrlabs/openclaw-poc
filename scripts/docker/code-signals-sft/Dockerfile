# syntax=docker/dockerfile:1.7
# ROCm PyTorch base for Code Signals LoRA SFT on AMD GPUs.
# Training script is mounted from the repo at runtime (see .github/workflows/code-signals.yml).
# Match host ROCm 7.x (gfx1100 / RX 7900 XTX). Override if your driver stack differs.
ARG ROCM_PYTORCH_IMAGE=rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.8.0
FROM ${ROCM_PYTORCH_IMAGE}

WORKDIR /opt/code-signals-sft

COPY requirements-sft.txt /opt/code-signals-sft/requirements-sft.txt
RUN pip install --no-cache-dir -r /opt/code-signals-sft/requirements-sft.txt

# Default command is overridden by workflow; keep a helpful smoke entrypoint.
ENTRYPOINT ["python3"]
