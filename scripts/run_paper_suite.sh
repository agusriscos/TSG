#!/usr/bin/env bash
# Nightly / smoke audit suite: Bloque 3 → 1 → 2 → 4 → 5 + packaging.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROFILE="paper"
OUTPUT_ROOT="${ROOT}/outputs"
WORKERS="${WORKERS:-}"
RESUME=0
LANG_FLAG="en"
SMOKE=0

usage() {
  cat <<'EOF'
Usage: scripts/run_paper_suite.sh [options]

Runs the manuscript audit stack in order:
  3 hard frontiers → 1 information → 2 surrogates → 4 invariance → 5 phase maps

Options:
  --profile {paper,dev}   Audit profile (default: paper)
  --smoke                 Fast end-to-end chain (R=4, N=200, tiny phase-map grid)
  --output-root DIR       Root under which reports/ and paper_bundle/ are written
                          (default: <repo>/outputs)
  --resume                Skip blocks that already have config_snapshot.json + metrics.json
  --workers N             Parallel workers (default: max(1, cpu_count-1))
  --lang {en,es}          Figure language (default: en)

Environment:
  WORKERS                 Same as --workers when the flag is omitted
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME=1; shift ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --lang) LANG_FLAG="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --smoke) SMOKE=1; PROFILE="dev"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "${PROFILE}" != "paper" && "${PROFILE}" != "dev" ]]; then
  echo "Invalid --profile ${PROFILE!r} (expected paper|dev)" >&2
  exit 2
fi

# Resolve relative output roots against the repo root.
if [[ "${OUTPUT_ROOT}" != /* ]]; then
  OUTPUT_ROOT="${ROOT}/${OUTPUT_ROOT}"
fi

if [[ -z "${WORKERS}" ]]; then
  WORKERS="$(python - <<'PY'
import os
print(max(1, (os.cpu_count() or 1) - 1))
PY
)"
fi

REPORTS="${OUTPUT_ROOT}/reports"
BUNDLE="${OUTPUT_ROOT}/paper_bundle"
LOG="${BUNDLE}/run_log.txt"

mkdir -p "${BUNDLE}/figures" "${BUNDLE}/tables" \
  "${REPORTS}/phase2_hard_frontiers" \
  "${REPORTS}/phase2_information" \
  "${REPORTS}/phase2_surrogates" \
  "${REPORTS}/phase2_invariance" \
  "${REPORTS}/phase2_phase_maps"

{
  echo "===== paper suite start $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  echo "ROOT=${ROOT}"
  echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "PROFILE=${PROFILE}"
  echo "SMOKE=${SMOKE}"
  echo "WORKERS=${WORKERS}"
  echo "LANG=${LANG_FLAG}"
  echo "RESUME=${RESUME}"
} | tee -a "${LOG}"

block_done() {
  local dir="$1"
  [[ -f "${dir}/config_snapshot.json" && -f "${dir}/metrics.json" ]]
}

run_block() {
  local name="$1"
  local out_dir="$2"
  shift 2
  echo "" | tee -a "${LOG}"
  echo "----- ${name} -----" | tee -a "${LOG}"
  if [[ "${RESUME}" -eq 1 ]] && block_done "${out_dir}"; then
    echo "SKIP ${name} (resume: config_snapshot.json + metrics.json present)" | tee -a "${LOG}"
    return 0
  fi
  local t0 t1
  t0="$(date +%s)"
  echo "CMD: $*" | tee -a "${LOG}"
  "$@"
  t1="$(date +%s)"
  echo "DONE ${name} in $((t1 - t0))s → ${out_dir}" | tee -a "${LOG}"
}

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  PY="$(command -v python)"
fi

HF_OUT="${REPORTS}/phase2_hard_frontiers"
INFO_OUT="${REPORTS}/phase2_information"
SURR_OUT="${REPORTS}/phase2_surrogates"
INV_OUT="${REPORTS}/phase2_invariance"
PM_OUT="${REPORTS}/phase2_phase_maps"

# Smoke / fast overrides: keep the full chain but shrink cost.
# Must keep Bloque 1 sizes aligned with Bloque 3 so the ensemble cache hits.
EXTRA_HF=()
EXTRA_INFO=()
EXTRA_SURR=()
EXTRA_INV=()
EXTRA_PM=()
if [[ "${SMOKE}" -eq 1 ]]; then
  # Keep methods aligned with Bloque 1 so the ensemble cache has zero misses.
  EXTRA_HF=(--n-realizations 4 --n-steps 200)
  EXTRA_INFO=(--n-realizations 4 --n-steps 200)
  EXTRA_SURR=(--quick --n-realizations 4 --n-steps 200)
  EXTRA_INV=(--quick --n-realizations 4 --n-steps 200)
  EXTRA_PM=(--quick --n-realizations 4)
fi

run_block "bloque3_hard_frontiers" "${HF_OUT}" \
  "${PY}" "${ROOT}/scripts/run_phase1_audit.py" \
  --config "${ROOT}/configs/phase2_hard_frontiers.yaml" \
  --profile "${PROFILE}" --lang "${LANG_FLAG}" --workers "${WORKERS}" \
  --output-dir "${HF_OUT}" \
  ${EXTRA_HF[@]+"${EXTRA_HF[@]}"}

run_block "bloque1_information" "${INFO_OUT}" \
  "${PY}" "${ROOT}/scripts/run_information_audit.py" \
  --config "${ROOT}/configs/phase2_hard_frontiers.yaml" \
  --profile "${PROFILE}" --lang "${LANG_FLAG}" --workers "${WORKERS}" \
  --cache-dir "${HF_OUT}/ensemble_cache" \
  --output-dir "${INFO_OUT}" \
  ${EXTRA_INFO[@]+"${EXTRA_INFO[@]}"}

run_block "bloque2_surrogates" "${SURR_OUT}" \
  "${PY}" "${ROOT}/scripts/run_surrogate_audit.py" \
  --config "${ROOT}/configs/phase2_surrogates.yaml" \
  --profile "${PROFILE}" --lang "${LANG_FLAG}" --workers "${WORKERS}" \
  --output-dir "${SURR_OUT}" \
  ${EXTRA_SURR[@]+"${EXTRA_SURR[@]}"}

run_block "bloque4_invariance" "${INV_OUT}" \
  "${PY}" "${ROOT}/scripts/run_invariance_audit.py" \
  --config "${ROOT}/configs/phase2_invariance.yaml" \
  --profile "${PROFILE}" --lang "${LANG_FLAG}" --workers "${WORKERS}" \
  --output-dir "${INV_OUT}" \
  ${EXTRA_INV[@]+"${EXTRA_INV[@]}"}

run_block "bloque5_phase_maps" "${PM_OUT}" \
  "${PY}" "${ROOT}/scripts/run_phase_maps_audit.py" \
  --config "${ROOT}/configs/phase2_phase_maps.yaml" \
  --profile "${PROFILE}" --lang "${LANG_FLAG}" --workers "${WORKERS}" \
  --output-dir "${PM_OUT}" \
  ${EXTRA_PM[@]+"${EXTRA_PM[@]}"}

echo "" | tee -a "${LOG}"
echo "Packaging figures and tables into ${BUNDLE}" | tee -a "${LOG}"

copy_figs() {
  local src="$1"
  local prefix="$2"
  if [[ ! -d "${src}" ]]; then
    return 0
  fi
  # Keep the original basename so .pdf/.svg/.png of the same figure share a stem.
  while IFS= read -r -d '' f; do
    local base
    base="$(basename "${f}")"
    cp -f "${f}" "${BUNDLE}/figures/${prefix}_${base}"
  done < <(find "${src}" -type f \( -name '*.pdf' -o -name '*.svg' -o -name '*.png' \) -print0 | sort -z)
}

copy_tables() {
  local src="$1"
  local prefix="$2"
  if [[ ! -d "${src}" ]]; then
    return 0
  fi
  while IFS= read -r -d '' f; do
    local base
    base="$(basename "${f}")"
    cp -f "${f}" "${BUNDLE}/tables/${prefix}_${base}"
  done < <(find "${src}" -type f -name '*.csv' -print0 | sort -z)
}

copy_figs "${HF_OUT}/figures" "b3_hard"
copy_figs "${INFO_OUT}" "b1_info"
copy_figs "${INFO_OUT}/figures" "b1_info"
copy_figs "${SURR_OUT}/figures" "b2_surr"
copy_figs "${INV_OUT}/figures" "b4_inv"
copy_figs "${PM_OUT}/figures" "b5_phase"

copy_tables "${HF_OUT}/data/tables" "b3"
copy_tables "${INFO_OUT}/data/tables" "b1"
copy_tables "${SURR_OUT}/data/tables" "b2"
copy_tables "${INV_OUT}/data/tables" "b4"
copy_tables "${PM_OUT}/data/tables" "b5"

mkdir -p "${BUNDLE}/notes"
if [[ -f "${SURR_OUT}/arma_iaaft_diagnosis.md" ]]; then
  cp -f "${SURR_OUT}/arma_iaaft_diagnosis.md" "${BUNDLE}/notes/b2_arma_iaaft_diagnosis.md"
fi

echo "===== paper suite finished $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" | tee -a "${LOG}"
echo "Bundle ready: ${BUNDLE}"
