#!/usr/bin/env bash
# Weekly full regression entry point.
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash ci/sail/run_weekly_tests.sh [--dry-run|--run]

Select cases.csv and run CPU, then CUDA, using run_tests.sh.
The full suite contains 9996 methods (5382 CPU, 4614 PPU/CUDA).
No argument or --dry-run previews commands without starting builds/tests.
--run executes both profiles in the foreground and returns a nonzero
exit code if either profile fails. The CUDA profile uses PPU by default.
The CI scheduler must trigger this script weekly; it does not schedule itself.

SAIL_ENV_FILE defaults to the local ppu_env.sh. The SDK, toolchain,
Python, and device environment must already be configured. Set
SAIL_ENV_FILE='' to use the launcher's standard TFCI environment instead.
SAIL_CUDA_VISIBLE_DEVICES selects the PPU device (default: 0).
SAIL_OUTPUT_ROOT sets the base results directory (default:
<source-root>/build_output/sail_final). Each run creates a unique weekly
subdirectory with separate CPU/CUDA logs and test artifacts.
SAIL_ONLY_TARGET and SAIL_TARGETS_FILE are cleared so the full suite runs.
HELP
}

if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi
mode="${1:---dry-run}"
case "$mode" in
  --run|--dry-run) ;;
  -h|--help) usage; exit 0 ;;
  *) printf 'Unknown argument: %s\n' "$mode" >&2; usage >&2; exit 2 ;;
esac

script_dir="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
repo_root="$(realpath "$script_dir/../..")"
manifest="$script_dir/cases.csv"
export SAIL_ENV_FILE="${SAIL_ENV_FILE-$script_dir/ppu_env.sh}"
unset SAIL_ONLY_TARGET SAIL_TARGETS_FILE

if [[ "$mode" == --run ]]; then
  # realpath -m is GNU-only, so create the base directory first and then resolve it.
  output_base="${SAIL_OUTPUT_ROOT:-$repo_root/build_output/sail_final}"
  mkdir -p "$output_base"
  output_base="$(realpath "$output_base")"
  job_dir="$output_base/weekly_$(date +%Y%m%d-%H%M%S)-$$"
  printf 'Weekly results root: %s\n' "$job_dir"
else
  printf '%s\n' 'Weekly preview only: no build, test, or output files.'
fi

overall=0
for profile in cpu cuda; do
  if [[ "$mode" == --run ]]; then
    export SAIL_OUTPUT_ROOT="$job_dir/$profile/ci_output"
    export SAIL_RUN_DIR="$job_dir/$profile/tests"
  fi
  printf '\nWeekly profile: %s\n' "$profile"
  if bash "$script_dir/run_tests.sh" --profile "$profile" --case-list "$manifest" "$mode"; then
    printf 'Weekly profile %s completed successfully.\n' "$profile"
  else
    result=$?
    printf 'Weekly profile %s failed with exit code %s.\n' "$profile" "$result" >&2
    overall=1
  fi
done
printf '\nWeekly overall exit code: %s\n' "$overall"
exit "$overall"
