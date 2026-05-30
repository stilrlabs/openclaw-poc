#!/usr/bin/env bash
# Merge synthetic LoRA, convert to GGUF, and assemble a portable package (sha + run number).
set -euo pipefail

LORA_DIR="${LORA_DIR:?LORA_DIR required (artifact root with adapter/)}"
OUT_DIR="${OUT_DIR:?OUT_DIR required}"
GIT_SHA="${GIT_SHA:?GIT_SHA required}"
RUN_NUMBER="${RUN_NUMBER:?RUN_NUMBER required}"
GGUF_OUTTYPE="${GGUF_OUTTYPE:-q8_0}"
LLAMA_CPP_GIT_REF="${LLAMA_CPP_GIT_REF:-master}"
ROCM_DEVICE_ID="${ROCM_DEVICE_ID:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/.artifacts/lora-gguf-work}"
MERGED_HF="${WORK_DIR}/merged-hf"
LLAMA_CPP_DIR="${WORK_DIR}/llama.cpp"

mkdir -p "${OUT_DIR}" "${WORK_DIR}"

echo "==> Merge LoRA into base HF snapshot"
python3 "${REPO_ROOT}/scripts/merge-lora-adapter.py" \
  --lora-dir "${LORA_DIR}" \
  --merged-dir "${MERGED_HF}" \
  --rocm-device-id "${ROCM_DEVICE_ID}"

MODEL_ID="$(python3 -c "
import json
from pathlib import Path
metrics = Path('${LORA_DIR}') / 'train-metrics.json'
print(json.loads(metrics.read_text(encoding='utf-8'))['model_id'])
")"

echo "==> Fetch llama.cpp (${LLAMA_CPP_GIT_REF}) for HF→GGUF conversion"
if [[ ! -d "${LLAMA_CPP_DIR}/.git" ]]; then
  git clone --depth 1 --branch "${LLAMA_CPP_GIT_REF}" https://github.com/ggml-org/llama.cpp.git "${LLAMA_CPP_DIR}"
else
  git -C "${LLAMA_CPP_DIR}" fetch --depth 1 origin "${LLAMA_CPP_GIT_REF}" || true
  git -C "${LLAMA_CPP_DIR}" checkout "${LLAMA_CPP_GIT_REF}" || true
fi

if [[ -f "${LLAMA_CPP_DIR}/requirements.txt" ]]; then
  python3 -m pip install --no-cache-dir -r "${LLAMA_CPP_DIR}/requirements.txt"
fi

PACKAGE_ID="openclaw-code-signals-${GIT_SHA}-run${RUN_NUMBER}"
GGUF_NAME="${PACKAGE_ID}-${GGUF_OUTTYPE}.gguf"
GGUF_PATH="${OUT_DIR}/${GGUF_NAME}"

echo "==> Convert merged HF snapshot to GGUF (${GGUF_OUTTYPE})"
python3 "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" "${MERGED_HF}" \
  --outfile "${GGUF_PATH}" \
  --outtype "${GGUF_OUTTYPE}"

cat > "${OUT_DIR}/manifest.json" <<EOF
{
  "package_id": "${PACKAGE_ID}",
  "git_sha": "${GIT_SHA}",
  "github_run_number": ${RUN_NUMBER},
  "base_model_id": "${MODEL_ID}",
  "gguf_file": "${GGUF_NAME}",
  "gguf_outtype": "${GGUF_OUTTYPE}",
  "llama_cpp_git_ref": "${LLAMA_CPP_GIT_REF}",
  "system_prompt": "You answer questions about this repository using grounded facts from extracted code and documentation artifacts."
}
EOF

cat > "${OUT_DIR}/Modelfile" <<EOF
# Ollama (after importing the GGUF): ollama create ${PACKAGE_ID} -f Modelfile
FROM ./${GGUF_NAME}
SYSTEM "You answer questions about this repository using grounded facts from extracted code and documentation artifacts."
EOF

cat > "${OUT_DIR}/README.txt" <<EOF
OpenClaw Code Signals merged model package
==========================================
Package ID: ${PACKAGE_ID}
Git SHA:    ${GIT_SHA}
Run:        ${RUN_NUMBER}
Base:       ${MODEL_ID}
GGUF:       ${GGUF_NAME} (${GGUF_OUTTYPE})

LM Studio
---------
1. LM Studio → Import Model → select ${GGUF_NAME}
2. Use the system prompt from manifest.json (or Modelfile)

Ollama (merged GGUF path)
-------------------------
Import ${GGUF_NAME} via your Ollama GGUF import flow, or merge locally and use FROM on the imported tag.

Source adapter artifact: synthetic-lora-${GIT_SHA}
EOF

echo "==> Package ready: ${OUT_DIR}"
ls -la "${OUT_DIR}"
