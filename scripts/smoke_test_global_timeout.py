#!/usr/bin/env python3
"""Smoke test for global wall-clock timeout in Speakeasy-X.

This test verifies that:
1. The patched winemu.py imports correctly
2. Global timeout works (limits total time across all runs)
3. Report is generated even after timeout
4. A small benign PE file completes normally within timeout
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEAKEASY_ROOT = Path(r"E:\Project\python\Speakeasy-X")

# Use any PE file available in the system for a quick test.
# The Windows calculator or notepad are good candidates.
TEST_PE_CANDIDATES = [
    Path(r"C:\Windows\System32\calc.exe"),
    Path(r"C:\Windows\System32\notepad.exe"),
    Path(r"C:\Windows\System32\mspaint.exe"),
]


def find_test_pe() -> Path | None:
    for p in TEST_PE_CANDIDATES:
        if p.exists():
            return p
    return None


def test_import() -> bool:
    """Test 1: Verify patched winemu.py imports correctly."""
    code = r"""
import sys
sys.path.insert(0, sys.argv[1])
import speakeasy
from speakeasy.windows.winemu import WindowsEmulator
# Verify time module is imported
import speakeasy.windows.winemu as wm
assert 'time' in dir(wm), "time module not imported in winemu"
print("IMPORT_OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(SPEAKEASY_ROOT)],
        capture_output=True, text=True, timeout=15,
        cwd=str(ROOT),
    )
    ok = "IMPORT_OK" in result.stdout
    print(f"  Test 1 (import): {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"    stderr: {result.stderr[:200]}")
    return ok


def test_global_timeout(test_pe: Path) -> bool:
    """Test 2: Verify global timeout limits total emulation time."""
    code = r"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[1])
target = Path(sys.argv[2])
emu_timeout = float(sys.argv[3])

import speakeasy
from speakeasy.config import get_default_config_dict

# Disable auto-mount to avoid scanning sibling files
speakeasy.Speakeasy._auto_mount_target_directory = lambda self, path: None

cfg = get_default_config_dict()
cfg["timeout"] = emu_timeout
cfg["max_api_count"] = 500
cfg["max_instructions"] = 100000
cfg.setdefault("analysis", {})["strings"] = False
cfg.setdefault("analysis", {})["coverage"] = False
cfg.setdefault("analysis", {})["memory_tracing"] = False

t0 = time.time()
se = speakeasy.Speakeasy(config=cfg)
mod = se.load_module(str(target))
se.run_module(mod, all_entrypoints=True, emulate_children=False)
elapsed = time.time() - t0

report = json.loads(se.get_json_report() or "{}")
eps = len(report.get("entry_points") or [])

# The key assertion: elapsed time should be close to emu_timeout, not
# emu_timeout * number_of_runs
print(json.dumps({
    "elapsed": round(elapsed, 2),
    "entry_points": eps,
    "emu_timeout": emu_timeout,
    "target": str(target.name),
}))
"""
    emu_timeout = 5.0
    result = subprocess.run(
        [sys.executable, "-c", code, str(SPEAKEASY_ROOT), str(test_pe), str(emu_timeout)],
        capture_output=True, text=True,
        timeout=int(emu_timeout * 3 + 10),  # generous outer timeout
        cwd=str(ROOT),
    )
    lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    if not lines:
        print(f"  Test 2 (global timeout): FAIL (no output)")
        print(f"    stderr: {result.stderr[:300]}")
        return False

    try:
        data = json.loads(lines[-1])
    except Exception as e:
        print(f"  Test 2 (global timeout): FAIL (bad JSON: {e})")
        print(f"    output: {lines[-1][:200]}")
        return False

    elapsed = data["elapsed"]
    eps = data["entry_points"]

    # With global timeout, elapsed should be <= emu_timeout + 2s (some overhead)
    # With OLD per-run timeout, elapsed could be emu_timeout * eps (much worse)
    threshold = emu_timeout + 3.0  # allow 3s overhead for init/teardown
    ok = elapsed <= threshold

    print(f"  Test 2 (global timeout): {'PASS' if ok else 'FAIL'}")
    print(f"    PE file: {data['target']}")
    print(f"    Entry points found: {eps}")
    print(f"    Elapsed: {elapsed:.2f}s (threshold: {threshold:.1f}s)")
    print(f"    Emu timeout: {emu_timeout}s")

    if not ok:
        print(f"    FAIL: elapsed {elapsed:.2f}s > threshold {threshold:.1f}s")
        print(f"    This suggests global timeout is NOT working (still per-run)")
    else:
        print(f"    OK: elapsed within budget, report has {eps} entry points")

    return ok


def test_report_after_timeout(test_pe: Path) -> bool:
    """Test 3: Verify report is generated even after timeout."""
    code = r"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[1])
target = Path(sys.argv[2])

import speakeasy
from speakeasy.config import get_default_config_dict

speakeasy.Speakeasy._auto_mount_target_directory = lambda self, path: None

cfg = get_default_config_dict()
cfg["timeout"] = 2  # Very short timeout to trigger global timeout quickly
cfg["max_api_count"] = 200
cfg["max_instructions"] = 50000
cfg.setdefault("analysis", {})["strings"] = False
cfg.setdefault("analysis", {})["coverage"] = False
cfg.setdefault("analysis", {})["memory_tracing"] = False

se = speakeasy.Speakeasy(config=cfg)
mod = se.load_module(str(target))
se.run_module(mod, all_entrypoints=True, emulate_children=False)

# After timeout, we should still get a report
report_json = se.get_json_report() or "{}"
report = json.loads(report_json)
has_sha256 = bool(report.get("sha256"))
has_eps = len(report.get("entry_points") or []) > 0
print(json.dumps({"has_entry_points": has_eps, "has_sha256": has_sha256, "sha256": report.get("sha256", ""), "entry_points": len(report.get("entry_points") or [])}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(SPEAKEASY_ROOT), str(test_pe)],
        capture_output=True, text=True,
        timeout=30,
        cwd=str(ROOT),
    )
    lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    if not lines:
        print(f"  Test 3 (report after timeout): FAIL (no output)")
        print(f"    stderr: {result.stderr[:300]}")
        return False

    try:
        data = json.loads(lines[-1])
    except Exception as e:
        print(f"  Test 3 (report after timeout): FAIL (bad JSON: {e})")
        return False

    ok = bool(data.get("sha256")) or data["has_entry_points"]  # Report should have at least some data
    print(f"  Test 3 (report after timeout): {'PASS' if ok else 'FAIL'}")
    print(f"    Report has sha256: {bool(data.get('sha256'))}")
    print(f"    Report has entry_points: {data['has_entry_points']} ({data['entry_points']} total)")
    return ok


def main() -> None:
    print("=" * 60)
    print("Speakeasy-X Global Timeout Smoke Test")
    print("=" * 60)

    test_pe = find_test_pe()
    if test_pe is None:
        print("ERROR: No test PE file found in system candidates:")
        for p in TEST_PE_CANDIDATES:
            print(f"  - {p}")
        sys.exit(1)

    print(f"\nUsing test PE: {test_pe}")
    print()

    results = []
    results.append(("Import", test_import()))
    results.append(("Global Timeout", test_global_timeout(test_pe)))
    results.append(("Report After Timeout", test_report_after_timeout(test_pe)))

    print()
    print("=" * 60)
    print("Results Summary:")
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        if not ok:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
