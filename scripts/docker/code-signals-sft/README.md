# Code Signals ROCm SFT container

Docker image for LoRA fine-tuning on synthetic training data produced by the Code Signals workflow.

Python deps are pinned in `requirements-sft.txt` (currently `transformers==5.9.0` and matching latest `peft`, `datasets`, `accelerate`, etc.).

## Host prerequisites

Run the checklist on the trainer machine:

```bash
./scripts/verify-trainer-host.sh
```

1. **GitHub Actions runner** registered as self-hosted with the **`trainer`** label (see `train-synthetic-lora` in `.github/workflows/code-signals.yml`).
2. **ROCm** installed on the host (`rocminfo` should list your GPU).
3. **Docker** for the Actions runner service (job uses `container:` — the runner pulls your image; workflow steps do not call `docker`). Runner user must reach `/var/run/docker.sock`; GPU devices pass through via `container.options`.
4. **VRAM** — roughly 4–8 GB for `Qwen/Qwen3.5-0.8B` LoRA smoke settings.
5. **`HF_TOKEN`** — add as a repository or organization secret for reliable Hugging Face Hub downloads (optional if the model is pre-cached on the runner).

## Local smoke (AMD host)

```bash
docker build -t openclaw-code-signals-sft:local scripts/docker/code-signals-sft

docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video \
  --ipc=host \
  --shm-size=16g \
  -v "$(pwd):/work" \
  -e HF_TOKEN \
  -e HUGGINGFACE_HUB_CACHE=/work/.cache/huggingface \
  openclaw-code-signals-sft:local \
  /work/scripts/code-signals-train-lora.py \
    --data /work/path/to/synthetic-chat-sft.jsonl \
    --output-dir /work/.artifacts/synthetic-lora/local-smoke \
    --max-steps 2 \
    --dry-run
```

## CI

The `train-synthetic-lora` job runs only on **workflow_dispatch** when you enable **Run LoRA training**. It runs inside the image set by **train_container_image** (default `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.8.0`), installs `requirements-sft.txt` in-container, then runs `scripts/code-signals-train-lora.py`. Override the image in the Actions UI without building a custom image locally.

`Dockerfile` in this directory is optional (local pre-baked image); CI does not `docker build` or `docker run`.
