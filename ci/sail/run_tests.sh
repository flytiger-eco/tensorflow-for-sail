#!/usr/bin/env bash
# Follow official PYCPP CI: TFCI -> setup.sh -> tfrun bazel test.
# Preview by default; only --run initializes official CI and starts builds/tests.
set -euo pipefail

# Official setup.sh locates the source root using $0, so make the entry point absolute first.
if [[ "$0" != /* ]]; then
  exec bash "$(realpath "$0")" "$@"
fi
script_dir="$(dirname "$0")"
repo_root="$(realpath "$script_dir/../..")"
python_bin="${SAIL_PYTHON:-python3}"
profile=cpu
mode=--dry-run
explicit_mode=
case_list="$script_dir/cases.csv"
target_list=

usage() {
  cat <<'HELP'
Usage: bash ci/sail/run_tests.sh [--profile cpu|cuda] [--dry-run|--run]
       [--case-list PATH] [--target-list PATH]

Preview is the default: no setup.sh, Bazel/Docker invocation, or test execution.
  bash ci/sail/run_tests.sh --profile cpu --dry-run
  bash ci/sail/run_tests.sh --profile cuda --dry-run
  bash ci/sail/run_tests.sh --profile cpu --case-list ci/sail/pr_targets.list --dry-run

Explicit execution reuses the official environment; TFCI defaults to py312 and the corresponding Linux platform:
  TFCI=py312,linux_x86 bash ci/sail/run_tests.sh --profile cpu --run
  TFCI=py312,linux_x86_cuda bash ci/sail/run_tests.sh --profile cuda --run

When running inside a configured container, append no_docker to TFCI to avoid creating the official tf container.
PPU must use its validated TFCI build options; this entry point does not configure or replace the PPU SDK.
SAIL_PYTHON selects the Python interpreter for manifest parsing on the host and in CI; the default is python3.
SAIL_ENV_FILE selects an environment adapter; for PPU, use the local ppu_env.sh from a container login shell.
PPU options default to the local ppu_config.sh; SAIL_PPU_CONFIG can override it. The legacy test tools directory is not required.
SAIL_RUN_DIR sets the output directory for this run; SAIL_CUDA_VISIBLE_DEVICES defaults to 0 for PPU.
SAIL_ONLY_TARGET reruns one listed target; SAIL_TARGETS_FILE selects a subset of targets to rerun.

cases.csv contains the full set of 9996 methods; pr_targets.list contains the 1996-method PR subset, also in CSV format.
Both share targets_cpu.list / targets_cuda.list. Select either with --case-list; do not pass the PR manifest to --target-list.
Run Python/absl methods grouped by target; no new tags or BUILD rules are needed.
Disable test caching, repeated runs, retries, and sharding for exact result reconciliation; retain other CI build options.
Requires the full TensorFlow source tree, ci/official, Bazel, and a compatible toolchain; execution may incur build costs.
After changing the environment or method set, use the current run's results to determine pass status.
HELP
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile|--case-list|--target-list)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        printf 'Missing value for argument %s\n' "$1" >&2
        exit 2
      fi
      case "$1" in
        --profile) profile="$2" ;;
        --case-list) case_list="$2" ;;
        --target-list) target_list="$2" ;;
      esac
      shift 2 ;;
    --run|--dry-run)
      if [[ -n "$explicit_mode" && "$explicit_mode" != "$1" ]]; then
        printf '%s\n' '--run and --dry-run are mutually exclusive' >&2
        exit 2
      fi
      mode="$1"
      explicit_mode="$1"
      shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

case "$profile" in
  cpu) prefix=linux_cpu; default_tfci=py312,linux_x86 ;;
  cuda) prefix=linux_cuda; default_tfci=py312,linux_x86_cuda ;;
  *) printf '%s\n' '--profile must be cpu or cuda' >&2; exit 2 ;;
esac
if [[ -n "${SAIL_ENV_FILE:-}" ]]; then
  source "$(realpath "$SAIL_ENV_FILE")"
fi
export TFCI="${TFCI:-$default_tfci}"
case_list="$(realpath "$case_list")"
target_list="$(realpath "${target_list:-$script_dir/targets_${profile}.list}")"
manifest_args=(--manifest "$case_list" --targets "$target_list" --profile "$profile")
plan="$("$python_bin" "$script_dir/manifest.py" plan "${manifest_args[@]}")"
if [[ -n "${SAIL_ONLY_TARGET:-}" ]] && ! grep -Fxq "$SAIL_ONLY_TARGET" <<< "$(cut -f1 <<< "$plan")"; then
  printf 'SAIL_ONLY_TARGET is not in the manifest for this profile: %s\n' "$SAIL_ONLY_TARGET" >&2
  exit 2
fi

if [[ -n "${SAIL_TARGETS_FILE:-}" ]]; then
  SAIL_TARGETS_FILE="$(realpath "$SAIL_TARGETS_FILE")"
  if [[ ! -f "$SAIL_TARGETS_FILE" ]]; then printf '%s\n' 'Target subset file does not exist' >&2; exit 2; fi
  while IFS= read -r chosen; do
    [[ -z "$chosen" || "$chosen" == \#* ]] && continue
    if ! grep -Fxq "$chosen" <<< "$(cut -f1 <<< "$plan")"; then
      printf 'Target to rerun is not in the current manifest: %s\n' "$chosen" >&2; exit 2
    fi
  done < "$SAIL_TARGETS_FILE"
fi

# Preserve pycpp.sh build options; replace only the broad test suite with filters and external lists.
read -r -a platform_options <<< "${SAIL_BAZEL_PLATFORM_ARGS---@local_config_cuda//cuda:override_include_cuda_libs=true --config=hermetic_cuda_umd}"
test_options=()
if [[ ${#platform_options[@]} -gt 0 ]]; then
  test_options+=("${platform_options[@]}")
fi
test_options+=(
  "--config=${prefix}_pycpp_test_filters"
  --test_sharding_strategy=disabled
  --runs_per_test=1
  --flaky_test_attempts=1
  --cache_test_results=no
)

if [[ "$mode" == --dry-run ]]; then
  printf 'Preview only, TFCI=%s; official setup.sh resolves build options when --run is used.\n' "$TFCI"
  index=0
  while IFS=$'\t' read -r target filter; do
    index=$((index + 1))
    if [[ -n "${SAIL_ONLY_TARGET:-}" && "$target" != "$SAIL_ONLY_TARGET" ]]; then continue; fi
    if [[ -n "${SAIL_TARGETS_FILE:-}" ]] && ! grep -Fxq "$target" "$SAIL_TARGETS_FILE"; then continue; fi
    printf '\nTarget %d: %s\n' "$index" "$target"
    printf 'tfrun bazel ${TFCI_BAZEL_BAZELRC_ARGS} test ${TFCI_BAZEL_COMMON_ARGS} '
    printf '%q ' "${test_options[@]}" "--test_filter=$filter" \
      "--target_pattern_file=<output_dir>/$index/targets.list" \
      "--build_event_json_file=<output_dir>/$index/bep.jsonl"
    printf '\n'
  done <<< "$plan"
  exit 0
fi

cd "$repo_root"
# setup.sh resolves TFCI_OUTPUT_DIR with realpath before creating it, which only tolerates
# a missing final component; the nested per-profile directories must exist beforehand.
if [[ -n "${TFCI_OUTPUT_DIR:-}" ]]; then
  mkdir -p "$TFCI_OUTPUT_DIR"
fi
source "$repo_root/ci/official/utilities/setup.sh"
if [[ "$TFCI_BAZEL_TARGET_SELECTING_CONFIG_PREFIX" != "$prefix" ]]; then
  printf 'TFCI selected %s, which does not match --profile %s\n' \
    "$TFCI_BAZEL_TARGET_SELECTING_CONFIG_PREFIX" "$profile" >&2
  exit 2
fi
if [[ "${TFCI_PYCPP_SWAP_TO_BUILD_ENABLE:-}" == 1 || "${TFCI_WHL_NUMPY_VERSION:-2}" != 2 ]]; then
  printf '%s\n' 'This entry point does not support build-only or numpy1 configurations.' >&2
  exit 2
fi
# Prevent common options from reintroducing broad target patterns such as //tensorflow/....
for value in $TFCI_BAZEL_COMMON_ARGS; do
  case "$value" in
    --|//*|@*//*|*_pycpp_test|*_wheel_test)
      printf 'Common options must not contain targets or full test suites: %s\n' "$value" >&2
      exit 2 ;;
  esac
done

output_dir="${SAIL_RUN_DIR:-$TFCI_OUTPUT_DIR/sail_final/${profile}_$(date +%Y%m%d-%H%M%S)-$$}"
mkdir -p "$output_dir"
printf '%s\n' "$output_dir" > "$TFCI_OUTPUT_DIR/latest_${profile}_run.txt"
printf '%s\n' "$plan" > "$output_dir/selection.tsv"
overall=0
index=0
# Use shell word splitting for TFCI options, as in official scripts; disable globbing to prevent filename expansion.
set -f
while IFS=$'\t' read -r target filter; do
  index=$((index + 1))
  if [[ -n "${SAIL_ONLY_TARGET:-}" && "$target" != "$SAIL_ONLY_TARGET" ]]; then continue; fi
  if [[ -n "${SAIL_TARGETS_FILE:-}" ]] && ! grep -Fxq "$target" "$SAIL_TARGETS_FILE"; then continue; fi
  group_dir="$output_dir/$index"
  bazel_status=0
  mkdir -p "$group_dir"
  printf '%s\n' "$target" > "$group_dir/targets.list"
  if tfrun bazel $TFCI_BAZEL_BAZELRC_ARGS test $TFCI_BAZEL_COMMON_ARGS \
      "${test_options[@]}" \
      "--test_filter=$filter" \
      "--target_pattern_file=$group_dir/targets.list" \
      "--build_event_json_file=$group_dir/bep.jsonl"; then
    bazel_status=0
  else
    bazel_status=$?
    overall=1
  fi
  # BEP file:// URIs refer to the execution environment; verify through tfrun in that same environment.
  if ! tfrun "$python_bin" "$script_dir/manifest.py" verify "${manifest_args[@]}" \
      --target "$target" --bep "$group_dir/bep.jsonl" \
      --bazel-exit-code "$bazel_status" --output "$group_dir/verification.json"; then
    overall=1
  fi
done <<< "$plan"
printf 'Results directory: %s; overall exit code: %s\n' "$output_dir" "$overall"
exit "$overall"
