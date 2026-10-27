#!/usr/bin/env python3
"""Parse external allowlists, build method filters, and verify Bazel BEP and method-level results.

This tool uses only the Python standard library. It neither imports TensorFlow nor starts Bazel.
The plan mode is read-only; verify reads completed execution results and writes a verification report.
Only Python/absl tests are supported; this filter syntax does not apply to C++ GoogleTest.
"""

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
IDENTIFIER = re.compile(r"[A-Za-z_]\w*", re.ASCII)
LABEL = re.compile(r"//tensorflow/[A-Za-z0-9_/.-]+:[A-Za-z0-9_.+-]+")
FIELDS = ("module", "case_id", "cpu_target", "cuda_target", "baseline_status",
          "baseline_log_sha256")


def checked_label(label):
    if not LABEL.fullmatch(label) or ".." in label:
        raise ValueError("Only explicit existing test labels are accepted; target patterns are not allowed: " + label)
    return label


def load_groups(manifest, target_list, profile):
    """Require an exact match between targets and method mappings to prevent extra or missing tests."""
    for path in (manifest, target_list):
        if ROOT not in path.resolve().parents:
            raise ValueError("Manifests must be inside the source tree to match the official Docker mounts")
    targets = []
    for raw in target_list.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            targets.append(checked_label(line))
    if not targets or len(targets) != len(set(targets)):
        raise ValueError("Target list is empty or contains duplicate labels")
    groups = collections.OrderedDict()
    seen = set()
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Method manifest CSV header does not match the expected schema")
        for number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Incorrect number of columns on CSV line {}".format(number))
            case_id = row["case_id"]
            if case_id in seen:
                raise ValueError("Duplicate logical test case: " + case_id)
            seen.add(case_id)
            parts = case_id.split("::")
            if len(parts) != 3:
                raise ValueError("Test case must use script::class::method format: " + case_id)
            source, class_name, method = parts
            if (not IDENTIFIER.fullmatch(class_name)
                    or not IDENTIFIER.fullmatch(method)
                    or not method.startswith("test")
                    or method in ("test_session", "test_scope")):
                raise ValueError("Unsupported method identifier: " + case_id)
            path = (ROOT / source).resolve()
            if (not source.startswith("tensorflow/") or path.suffix != ".py"
                    or ROOT / "tensorflow" not in path.parents or not path.is_file()):
                raise ValueError("Valid test source file not found: " + source)
            if (row["baseline_status"] != "PASSED"
                    or not re.fullmatch(r"[a-f0-9]{64}", row["baseline_log_sha256"])):
                raise ValueError("Entry must include a historical PASS status and log digest: " + case_id)
            if not row["cpu_target"] and not row["cuda_target"]:
                raise ValueError("Test case has no execution mapping: " + case_id)
            for field in ("cpu_target", "cuda_target"):
                if row[field]:
                    checked_label(row[field])
            target = row[profile + "_target"]
            if not target:
                continue
            selector = class_name + "." + method
            group = groups.setdefault(target, [])
            if selector in group:
                raise ValueError("Duplicate method within the same target: " + selector)
            group.append(selector)
    if set(targets) != set(groups):
        raise ValueError("Target list does not match the method mappings for this profile")
    return collections.OrderedDict((target, groups[target]) for target in targets), len(seen)


def method_filter(selectors):
    # absl splits TESTBRIDGE_TEST_ONLY with shlex into multiple unittest -k patterns (OR semantics).
    # The *. prefix preserves class/method boundaries so testFloat does not also select testFloatGPU.
    return " ".join("*." + selector for selector in selectors)


def local_artifact(outputs, name):
    matches = [item for item in outputs if item.get("name") == name]
    if len(matches) != 1:
        raise ValueError("Missing or duplicate BEP artifact: " + name)
    uri = urlparse(matches[0].get("uri", ""))
    if uri.scheme != "file" or uri.netloc not in ("", "localhost"):
        raise ValueError("A downloaded local {} is required; missing remote artifacts do not count as a pass".format(name))
    return Path(unquote(uri.path))


def check_execution(target, expected, bep_path, bazel_exit_code):
    report = {
        "target": target, "bazel_exit_code": bazel_exit_code,
        "status": "INVALID_RESULT", "expected": list(expected),
        "observed": [], "missing": list(expected), "unexpected": [], "cases": {},
    }
    try:
        results = []
        summaries = []
        with bep_path.open(encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                event_id = event.get("id", {})
                for kind in ("testResult", "testSummary"):
                    if kind not in event_id:
                        continue
                    if event_id[kind].get("label") != target:
                        raise ValueError("BEP contains a test target outside the allowlist")
                    if kind == "testResult":
                        results.append(event.get(kind, {}))
                    else:
                        summaries.append(event.get(kind, {}))
        # The launcher explicitly disables retries, repeated runs, and sharding.
        if len(results) != 1 or len(summaries) != 1:
            raise ValueError("Expected 1 test action and 1 summary; got {}/{}. Tests may have been filtered out or not executed".format(
                len(results), len(summaries)))
        result = results[0]
        outputs = result.get("testActionOutput", [])
        log_path = local_artifact(outputs, "test.log")
        report['log'] = str(log_path)
        xml_path = local_artifact(outputs, "test.xml")
        report['xml'] = str(xml_path)
        root = ET.parse(xml_path).getroot()
        log = log_path.read_text(encoding="utf-8", errors="replace")
        states = collections.defaultdict(list)
        for case in root.iter("testcase"):
            class_name = case.get("classname", "").rsplit(".", 1)[-1]
            # Parameter descriptions and graph/eager subTest suffixes do not increase the logical method count.
            method = case.get("name", "").split(" ", 1)[0]
            selector = class_name + "." + method
            if case.find("failure") is not None or case.find("error") is not None:
                state = "FAILED"
            elif case.find("skipped") is not None or case.get("status") == "notrun":
                state = "SKIPPED"
            else:
                state = "PASSED"
            states[selector].append(state)
        observed = set(states)
        report.update({
            "observed": sorted(observed), "missing": sorted(set(expected) - observed),
            "unexpected": sorted(observed - set(expected)),
            "xml": str(xml_path), "log": str(log_path),
            "cases": {selector: ("FAILED" if "FAILED" in items else
                                  "SKIPPED" if "SKIPPED" in items else "PASSED")
                      for selector, items in states.items()},
        })
        if report["missing"] or report["unexpected"]:
            raise ValueError("Executed method set does not match the allowlist")
        counts = re.findall(r"^Ran (\d+) tests? in ", log, re.MULTILINE)
        if counts != [str(len(expected))]:
            raise ValueError("Executed test count does not match; log reports " + repr(counts))
        if any(p.get("name") == "EXPECTED_FAILURE" for p in root.iter("property")):
            raise ValueError("Expected failures do not count as genuine passes")
        if (bazel_exit_code != 0 or result.get("status") != "PASSED"
                or summaries[0].get("overallStatus") != "PASSED"
                or any(value != "PASSED" for value in report["cases"].values())
                or next(root.iter("failure"), None) is not None
                or next(root.iter("error"), None) is not None):
            report["status"] = "FAILED"
            report["message"] = "Bazel, methods, or subtest results did not all pass"
        else:
            report["status"] = "PASSED"
            report["message"] = "Bazel and method-level results passed; the executed set matches exactly"
    except (OSError, ValueError, ET.ParseError, TypeError, KeyError) as error:
        report["message"] = str(error)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "verify"))
    parser.add_argument("--profile", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--manifest", type=Path, default=HERE / "cases.csv")
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--bep", type=Path)
    parser.add_argument("--bazel-exit-code", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    target_list = args.targets or HERE / ("targets_" + args.profile + ".list")
    try:
        groups, logical_count = load_groups(args.manifest, target_list, args.profile)
        if args.mode == "plan":
            print("{} logical test cases; {} selects {} methods across {} targets.".format(
                logical_count, args.profile, sum(map(len, groups.values())), len(groups)), file=sys.stderr)
            for target, selectors in groups.items():
                print(target + "\t" + method_filter(selectors))
            return 0
        if (args.target not in groups or args.bep is None or args.output is None
                or args.bazel_exit_code is None):
            raise ValueError("verify requires a valid target, BEP file, Bazel exit code, and output path")
        expected = groups[args.target]
        report = check_execution(args.target, expected, args.bep, args.bazel_exit_code)
        report['profile'] = args.profile
        report['manifest_sha256'] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        for artifact in ('xml', 'log'):
            if artifact in report:
                source = Path(report[artifact])
                if not source.is_file():
                    continue
                destination = args.output.parent / ('test.' + artifact)
                if source.resolve() != destination.resolve():
                    shutil.copyfile(source, destination)
                report[artifact] = str(destination)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print("{} {}: {}".format(report["status"], args.target, report["message"]))
        return 0 if report["status"] == "PASSED" else 1
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
