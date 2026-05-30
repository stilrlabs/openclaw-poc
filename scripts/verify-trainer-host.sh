#!/usr/bin/env bash
# Verify this machine can run the train-synthetic-lora workflow job (runs-on: trainer).
set -euo pipefail

ROCM_PYTORCH_IMAGE="${ROCM_PYTORCH_IMAGE:-rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.8.0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAIL=0
WARN=0

pass() { echo "[ok]   $*"; }
warn() { echo "[warn] $*"; WARN=$((WARN + 1)); }
fail() { echo "[fail] $*"; FAIL=$((FAIL + 1)); }

echo "=== Trainer host verification ==="
echo "Image: ${ROCM_PYTORCH_IMAGE}"
echo ""

echo "--- GitHub Actions runner (label: trainer) ---"
if pgrep -f 'Runner.Listener' >/dev/null 2>&1; then
  pass "actions-runner process is running"
else
  fail "no Runner.Listener process — install/start a self-hosted runner with label trainer"
  echo "       https://docs.github.com/en/actions/hosting-your-own-runners"
fi
if command -v systemctl >/dev/null 2>&1; then
  if systemctl --user list-unit-files 'actions.runner*' 2>/dev/null | rg -q 'actions\.runner'; then
    pass "runner systemd user unit file present"
  elif systemctl list-unit-files 'actions.runner*' 2>/dev/null | rg -q 'actions\.runner'; then
    pass "runner systemd system unit file present"
  else
    warn "no actions.runner systemd unit found (runner may be started manually)"
  fi
fi

echo "--- ROCm / GPU ---"
if [[ -e /dev/kfd && -d /dev/dri ]]; then
  pass "/dev/kfd and /dev/dri present"
else
  fail "missing /dev/kfd or /dev/dri — ROCm GPU not exposed"
fi
if command -v rocminfo >/dev/null 2>&1 || [[ -x /opt/rocm/bin/rocminfo ]]; then
  ROCMINFO="$(command -v rocminfo 2>/dev/null || echo /opt/rocm/bin/rocminfo)"
  ROCMINFO_TEXT="$("${ROCMINFO}" 2>&1 || true)"
  if printf '%s\n' "${ROCMINFO_TEXT}" | rg -q 'Radeon RX|gfx11|gfx10'; then
    GPU_LINE="$(printf '%s\n' "${ROCMINFO_TEXT}" | rg -m1 'Marketing Name:.*Radeon' | sed 's/^[[:space:]]*//')"
    pass "rocminfo reports AMD discrete GPU (${GPU_LINE:-detected})"
  else
    warn "rocminfo ran but no obvious discrete GPU line — check output manually"
  fi
else
  warn "rocminfo not on PATH (workflow continues; Docker preflight is authoritative)"
fi
if command -v rocm-smi >/dev/null 2>&1 || [[ -x /opt/rocm/bin/rocm-smi ]]; then
  pass "rocm-smi available"
else
  warn "rocm-smi not found"
fi

echo "--- Docker ---"
if command -v docker >/dev/null 2>&1; then
  pass "docker: $(docker --version)"
  if docker info >/dev/null 2>&1; then
    pass "docker daemon reachable"
  else
    fail "docker daemon not reachable (start docker / add user to docker group)"
  fi
else
  fail "docker not installed"
fi
if id -nG | rg -qw 'docker'; then
  pass "user in docker group"
else
  warn "user not in docker group — may need sudo for docker run"
fi
if id -nG | rg -qw 'video|render'; then
  pass "user in video/render group (GPU device access)"
else
  warn "user not in video/render — container still uses --group-add=video"
fi

echo "--- ROCm PyTorch container GPU ---"
if docker image inspect "${ROCM_PYTORCH_IMAGE}" >/dev/null 2>&1; then
  pass "image present locally: ${ROCM_PYTORCH_IMAGE}"
else
  warn "image not pulled yet — run: docker pull ${ROCM_PYTORCH_IMAGE}"
fi
if docker image inspect "${ROCM_PYTORCH_IMAGE}" >/dev/null 2>&1; then
  if docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add=video \
    --ipc=host \
    "${ROCM_PYTORCH_IMAGE}" \
    python3 -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))" \
    2>/dev/null; then
    pass "PyTorch sees ROCm GPU inside container"
  else
    fail "PyTorch in container cannot use GPU — fix devices/driver or image tag"
  fi
fi

echo "--- SFT training image (optional) ---"
if docker image inspect openclaw-code-signals-sft:local >/dev/null 2>&1; then
  pass "openclaw-code-signals-sft:local already built"
else
  warn "SFT image not built — workflow runs: docker build -t openclaw-code-signals-sft:local scripts/docker/code-signals-sft"
fi

echo "--- Disk / memory ---"
AVAIL_GB="$(df -BG --output=avail "${REPO_ROOT}" | tail -1 | tr -dc '0-9')"
if [[ "${AVAIL_GB}" -ge 30 ]]; then
  pass "disk free at repo: ${AVAIL_GB}G (need ~30G+ for model cache + build)"
else
  warn "disk free at repo: ${AVAIL_GB}G — may be tight for HF cache + Docker layers"
fi
if command -v free >/dev/null 2>&1; then
  AVAIL_MEM="$(free -g | awk '/^Mem:/ {print $7}')"
  if [[ "${AVAIL_MEM}" -ge 8 ]]; then
    pass "available RAM: ~${AVAIL_MEM}G"
  else
    warn "available RAM ~${AVAIL_MEM}G — prefer 8G+ free for smoke LoRA"
  fi
fi

echo "--- Repo secrets (manual) ---"
warn "Set GitHub repo secret HF_TOKEN for Hub downloads (or pre-cache Qwen/Qwen3.5-0.8B)"

echo ""
if [[ "${FAIL}" -eq 0 ]]; then
  echo "Result: READY (warnings: ${WARN})"
  exit 0
fi
echo "Result: NOT READY — ${FAIL} failure(s), ${WARN} warning(s)"
exit 1
