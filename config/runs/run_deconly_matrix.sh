#!/usr/bin/env bash
# Decomposition-only ablation matrix: 3 models x 2 AJCC editions.
# Run from the repository root:  bash config/runs/run_deconly_matrix.sh [filter]
#
# The optional filter matches the config suffix, e.g.
#   bash config/runs/run_deconly_matrix.sh llama3   -> the LLaMA lane
#   bash config/runs/run_deconly_matrix.sh 9        -> AJCC 9th only
#
# Credentials come from the environment (see config/runs/README.md); the tracked
# config files keep placeholder values.
set -u
cd "$(dirname "$0")/../.."

# Detach stdin so a closed terminal cannot hand a bad file descriptor to python.
exec < /dev/null

if   [ -n "${PY:-}" ] && command -v "$PY" >/dev/null 2>&1; then :
elif [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"
elif [ -x "venv/bin/python"  ]; then PY="venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then PY="python3"
else PY="python"; fi
echo "Interpreter: $PY  ($("$PY" --version 2>&1))"

IN="${IN:-input/synthetic_single_excel_ajcc_8th.xlsx}"   # override: IN=... bash ...
TS="$(date +%Y%m%d)"
TRY="${TRY:-try1}"
FILTER="${1:-}"

runs=(
  "deconly_8_llama3:llama3-70b-deconly-ajcc8"
  "deconly_8_phi4:phi4-deconly-ajcc8"
  "deconly_8_gpt4o:gpt4o-deconly-ajcc8"
  "deconly_9_llama3:llama3-70b-deconly-ajcc9"
  "deconly_9_phi4:phi4-deconly-ajcc9"
  "deconly_9_gpt4o:gpt4o-deconly-ajcc9"
)

mkdir -p output
for r in "${runs[@]}"; do
  cfg="${r%%:*}"; tag="${r##*:}"
  if [ -n "$FILTER" ] && [[ "$cfg" != *"$FILTER"* ]]; then continue; fi
  cfgfile="config/runs/tnm_config_${cfg}.yaml"
  out="output/${TS}_${tag}-${TRY}"
  log="output/${TS}_${tag}-${TRY}.runlog"
  if [ ! -f "$cfgfile" ]; then echo "!! missing $cfgfile - skip"; continue; fi
  echo "==================================================================="
  echo "==== $(date '+%F %T')  START  ${cfg}  ->  ${out}"
  echo "==================================================================="
  "$PY" run_workflow.py --i "$IN" --o "$out" --config "$cfgfile" < /dev/null > "$log" 2>&1
  rc=$?          # capture before any command substitution clobbers $?
  # Count from the runlog, not from "${out}.log": the application log is a rotating
  # handler that rolls over at 10 MB, so a long run leaves only its tail there and the
  # earlier cases move to "${out}.log.1". The runlog is the full, unrotated record.
  done_cases=$(grep -c "processing completed successfully" "$log" 2>/dev/null || echo 0)
  echo "==== $(date '+%F %T')  END    ${cfg}   (exit ${rc}, ${done_cases} cases completed)"
  if [ "$rc" -ne 0 ]; then
    echo "!! ${cfg} exited non-zero (${rc}). Last lines of ${log}:"
    tail -n 15 "$log"
    if [ "${STOP_ON_ERROR:-1}" = "1" ]; then
      echo "!! Stopping so the failure is not buried by later runs."
      echo "!! Re-run this configuration alone, or set STOP_ON_ERROR=0 to continue past failures."
      exit "$rc"
    fi
  fi
done
echo "ALL DONE: $(date '+%F %T')"
