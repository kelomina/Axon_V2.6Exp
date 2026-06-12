#!/usr/bin/env python3
"""Offline Speakeasy-X API sequence dataset extractor.

This script separates slow dynamic sequence extraction from model training.
It reads one or more prediction/manifest CSV files, runs Speakeasy-X once per
sample, and writes resumable feature rows plus per-sample reports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIR = ROOT / "reports" / "hard_family_finetune" / "clean_hyperparam_search"
DEFAULT_INPUTS = [
    SEARCH_DIR / "baseline_train_predictions_threshold053_current.csv",
    SEARCH_DIR / "baseline_val_predictions_threshold053_current.csv",
    SEARCH_DIR / "baseline_test_predictions_threshold053_current.csv",
]
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "speakeasy_sequence_dataset"
DEFAULT_SPEAKEASY_ROOT = Path(r"E:\Project\python\Speakeasy-X")
HELPER_DIR = SEARCH_DIR
if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))

import run_speakeasy_feature_probe as speakeasy_probe
import run_speakeasy_sequence_probe as sequence_probe


SEQUENCE_OUTPUT_COLUMNS = [
    "speakeasy_error",
    "top_api_names",
    "top_event_names",
    "extract_status",
    "extract_elapsed_seconds",
    "sample_timeout",
    "emu_timeout",
    "disable_auto_mount",
    "report_path",
    "api_sequence",
    "category_sequence",
    "event_sequence",
    "arg_sequence",
    "category_api_sequence",
    "sequence_event_count",
]


def read_rows(path: Path, annotate_input_csv: bool = True) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if annotate_input_csv:
        for row in rows:
            row["_input_csv"] = str(path)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    if needs_header:
        header = list(fieldnames or row.keys())
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
    missing_columns = [key for key in row if key not in header]
    if missing_columns:
        raise ValueError(
            f"Feature CSV schema is missing columns {missing_columns}. "
            f"Create a new output directory or extend feature_fieldnames()."
        )
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def row_key(row: dict) -> str:
    source_path = str(Path(row.get("source_path") or "").resolve()).lower()
    cache_path = str(Path(row.get("cache_path") or "").resolve()).lower() if row.get("cache_path") else ""
    label = str(row.get("label") or "")
    digest = hashlib.sha256(f"{source_path}|{cache_path}|{label}".encode("utf-8", errors="ignore")).hexdigest()
    return digest[:24]


def load_existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row.get("sample_id", "") for row in csv.DictReader(handle) if row.get("sample_id")}


def existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return read_rows(path, annotate_input_csv=False)


def source_exists(row: dict) -> bool:
    source = Path(row.get("source_path") or "")
    if not source.exists():
        return False
    cache_path = row.get("cache_path")
    return not cache_path or Path(cache_path).exists()


def select_rows(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    selected = []
    seen: set[str] = set()
    for row in rows:
        if args.split and str(row.get("split") or "") not in set(args.split):
            continue
        if args.label is not None and str(row.get("label") or "") != str(args.label):
            continue
        if not source_exists(row):
            continue
        sample_id = row_key(row)
        if sample_id in seen:
            continue
        seen.add(sample_id)
        enriched = dict(row)
        enriched["sample_id"] = sample_id
        enriched["probe_split"] = row.get("split") or "dataset"
        enriched["probe_role"] = "offline_dataset"
        enriched["probe_score_column"] = ""
        enriched["probe_score"] = ""
        selected.append(enriched)
    selected.sort(key=lambda item: (item.get("split", ""), item.get("label", ""), item["sample_id"]))
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def completed_enough(row: dict) -> bool:
    status = row.get("extract_status") or ""
    return status in {"success", "timeout_partial_report", "timeout_no_report", "unsupported_dotnet", "error", "worker_timeout"}


def classify_status(features: dict, report_path: Path) -> str:
    error = str(features.get("speakeasy_error") or "").lower()
    if "timeout" in error and report_path.exists() and report_path.stat().st_size > 0:
        return "timeout_partial_report"
    if "timeout" in error:
        return "timeout_no_report"
    if int(features.get("error_not_supported_dotnet") or 0):
        return "unsupported_dotnet"
    if int(features.get("speakeasy_success") or 0):
        return "success"
    return "error"


def empty_sequence() -> dict:
    return {
        "api_tokens": [],
        "category_tokens": [],
        "event_tokens": [],
        "arg_tokens": [],
        "mixed_tokens": [],
        "event_count_from_report": 0,
    }


def safe_sequence_from_report(report_path: Path, max_events: int) -> dict:
    if not report_path.exists() or report_path.stat().st_size == 0:
        return empty_sequence()
    try:
        return sequence_probe.sequence_from_report(report_path, max_events)
    except Exception:
        return empty_sequence()


def extract_one(row: dict, index: int, total: int, args: argparse.Namespace) -> dict:
    reports_dir = args.output_dir / "reports"
    report_path = reports_dir / f"{row['sample_id']}.json"
    if not args.overwrite and report_path.exists():
        sequence = safe_sequence_from_report(report_path, args.max_sequence_events)
    else:
        sequence = empty_sequence()

    worker_args = argparse.Namespace(
        save_reports=True,
        speakeasy_root=args.speakeasy_root,
        max_api_count=args.max_api_count,
        max_instructions=args.max_instructions,
        emu_timeout=args.emu_timeout,
        sample_timeout=args.sample_timeout,
        disable_auto_mount=args.disable_auto_mount,
        report_name=f"{row['sample_id']}.worker.json",
    )
    worker_row = dict(row)
    worker_row["probe_split"] = row.get("probe_split") or row.get("split") or "dataset"
    worker_row["probe_role"] = row.get("probe_role") or "offline_dataset"
    worker_report_path = reports_dir / f"{row['sample_id']}.worker.json"

    start = time.time()
    print(
        f"[{index}/{total}] sample_id={row['sample_id']} split={row.get('split', '')} "
        f"label={row.get('label', '')} {Path(row['source_path']).name}",
        flush=True,
    )
    features = speakeasy_probe.run_speakeasy_child(worker_row, index, worker_args, reports_dir)
    elapsed = time.time() - start
    if worker_report_path.exists() and worker_report_path != report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(worker_report_path, report_path)
    if report_path.exists():
        sequence = safe_sequence_from_report(report_path, args.max_sequence_events)

    out = dict(row)
    out.update(features)
    out["sample_id"] = row["sample_id"]
    out["extract_status"] = classify_status(features, report_path)
    out["extract_elapsed_seconds"] = f"{elapsed:.6f}"
    out["sample_timeout"] = str(args.sample_timeout)
    out["emu_timeout"] = str(args.emu_timeout)
    out["disable_auto_mount"] = str(bool(args.disable_auto_mount))
    out["report_path"] = str(report_path) if report_path.exists() else ""
    out["api_sequence"] = " ".join(sequence["api_tokens"])
    out["category_sequence"] = " ".join(sequence["category_tokens"])
    out["event_sequence"] = " ".join(sequence["event_tokens"])
    out["arg_sequence"] = " ".join(sequence["arg_tokens"])
    out["category_api_sequence"] = " ".join(sequence["mixed_tokens"])
    out["sequence_event_count"] = str(sequence["event_count_from_report"])
    print(
        f"    status={out['extract_status']} success={out.get('speakeasy_success')} "
        f"timeout={out.get('error_timeout')} api={out.get('api_call_count')} "
        f"seq={len(sequence['api_tokens'])} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return out


def extract_task(payload: tuple[dict, int, int, argparse.Namespace]) -> dict:
    row, index, total, args = payload
    return extract_one(row, index, total, args)


def finalize_output(row: dict, features: dict, report_path: Path, args: argparse.Namespace, elapsed: float) -> dict:
    sequence = empty_sequence()
    if report_path.exists():
        sequence = safe_sequence_from_report(report_path, args.max_sequence_events)

    out = dict(row)
    out.update(features)
    out["sample_id"] = row["sample_id"]
    out["extract_status"] = classify_status(features, report_path)
    out["extract_elapsed_seconds"] = f"{elapsed:.6f}"
    out["sample_timeout"] = str(args.sample_timeout)
    out["emu_timeout"] = str(args.emu_timeout)
    out["disable_auto_mount"] = str(bool(args.disable_auto_mount))
    out["report_path"] = str(report_path) if report_path.exists() else ""
    out["api_sequence"] = " ".join(sequence["api_tokens"])
    out["category_sequence"] = " ".join(sequence["category_tokens"])
    out["event_sequence"] = " ".join(sequence["event_tokens"])
    out["arg_sequence"] = " ".join(sequence["arg_tokens"])
    out["category_api_sequence"] = " ".join(sequence["mixed_tokens"])
    out["sequence_event_count"] = str(sequence["event_count_from_report"])
    return out


def zero_timeout_features(message: str) -> dict:
    return {
        "speakeasy_success": 0,
        "speakeasy_error": message,
        "error_not_supported_dotnet": 0,
        "error_not_supported_other": 0,
        "error_timeout": 1,
        "entry_point_count": 0,
        "event_count": 0,
        "api_call_count": 0,
        "unique_api_count": 0,
        "instr_total": 0,
        "runtime": 0,
        "top_level_error_count": 0,
        "error_count": 1,
        "static_string_count": 0,
        "memory_string_count": 0,
        "dropped_file_count": 0,
        "dynamic_code_segment_count": 0,
        "top_api_names": "",
        "top_event_names": "",
        "event_api_count": 0,
        "event_file_count": 0,
        "event_registry_count": 0,
        "event_network_count": 0,
        "event_process_count": 0,
        "event_memory_count": 0,
        "event_module_count": 0,
        "event_exception_count": 0,
        "api_network_count": 0,
        "api_process_count": 0,
        "api_injection_count": 0,
        "api_filesystem_count": 0,
        "api_registry_count": 0,
        "api_crypto_count": 0,
        "api_memory_count": 0,
        "elapsed_seconds": 0,
        "top_api_names": "",
        "top_event_names": "",
    }


def feature_fieldnames(selected: list[dict], existing: list[dict]) -> list[str]:
    fieldnames: list[str] = []
    for collection in (selected, existing):
        for row in collection:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    for key in speakeasy_probe.NUMERIC_FEATURES:
        if key not in fieldnames:
            fieldnames.append(key)
    for key in SEQUENCE_OUTPUT_COLUMNS:
        if key not in fieldnames:
            fieldnames.append(key)
    return fieldnames


def write_partial_summary(
    args: argparse.Namespace,
    feature_rows: list[dict],
    total: int,
    completed_now: int,
    start: float,
    pending_count: int,
) -> None:
    summary = summarize(feature_rows, args)
    summary["selected_rows"] = total
    summary["completed_rows"] = len(feature_rows)
    summary["pending_rows"] = max(0, pending_count)
    summary["newly_extracted_rows"] = completed_now
    summary["elapsed_seconds"] = time.time() - start
    summary["complete"] = False
    summary_path = args.output_dir / "summary_partial.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def speakeasy_child_env(args: argparse.Namespace) -> dict:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(args.speakeasy_root) + (os.pathsep + existing if existing else "")
    return env


def speakeasy_child_command(row: dict, args: argparse.Namespace, report_path: Path) -> list[str]:
    return [
        sys.executable,
        "-c",
        speakeasy_probe.CHILD_CODE,
        str(args.speakeasy_root),
        row["source_path"],
        str(args.max_api_count),
        str(args.max_instructions),
        str(args.emu_timeout),
        str(report_path),
        "1" if bool(args.disable_auto_mount) else "0",
    ]


def write_child_stderr(stderr: str, row: dict, reports_dir: Path) -> None:
    if not stderr:
        return
    reports_dir.mkdir(parents=True, exist_ok=True)
    stderr_path = reports_dir / f"{row['sample_id']}.stderr.txt"
    stderr_path.write_text(stderr, encoding="utf-8", errors="replace")


def recover_timeout_features(report_path: Path, message: str, elapsed: float) -> dict:
    partial = speakeasy_probe.report_feature_counts(report_path, message)
    features = partial if partial is not None else zero_timeout_features(message)
    features["elapsed_seconds"] = elapsed
    return features


def parse_child_features(proc: subprocess.Popen, stdout: str, row: dict, elapsed: float) -> dict:
    if proc.returncode != 0:
        features = zero_timeout_features(f"ChildReturnCode: {proc.returncode}")
        features["error_timeout"] = 0
        features["elapsed_seconds"] = elapsed
        return features
    lines = [line for line in (stdout or "").splitlines() if line.strip()]
    try:
        features = json.loads(lines[-1])
    except Exception as exc:
        preview = (stdout or "").strip().splitlines()[-3:]
        features = zero_timeout_features(f"BadChildJson: {type(exc).__name__}: {exc}; stdout_tail={preview}")
        features["error_timeout"] = 0
    features["elapsed_seconds"] = elapsed
    return features


def copy_report_if_available(worker_report_path: Path, report_path: Path) -> None:
    if worker_report_path.exists():
        report_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(worker_report_path, report_path)


def start_parallel_child(row: dict, index: int, total: int, args: argparse.Namespace, reports_dir: Path) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{row['sample_id']}.json"
    worker_report_path = reports_dir / f"{row['sample_id']}.worker.json"
    stdout_path = reports_dir / f"{row['sample_id']}.stdout.txt"
    stderr_path = reports_dir / f"{row['sample_id']}.stderr.txt"
    if worker_report_path.exists():
        worker_report_path.unlink()
    stdout_handle = stdout_path.open("w", encoding="utf-8", errors="replace")
    stderr_handle = stderr_path.open("w", encoding="utf-8", errors="replace")
    print(
        f"[{index}/{total}] sample_id={row['sample_id']} split={row.get('split', '')} "
        f"label={row.get('label', '')} {Path(row['source_path']).name}",
        flush=True,
    )
    try:
        proc = subprocess.Popen(
            speakeasy_child_command(row, args, worker_report_path),
            cwd=str(ROOT),
            env=speakeasy_child_env(args),
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
    except Exception:
        stdout_handle.close()
        stderr_handle.close()
        raise
    return {
        "proc": proc,
        "row": row,
        "index": index,
        "started_at": time.time(),
        "report_path": report_path,
        "worker_report_path": worker_report_path,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "stdout_handle": stdout_handle,
        "stderr_handle": stderr_handle,
    }


def finish_parallel_child(state: dict, args: argparse.Namespace, reports_dir: Path, timed_out: bool = False) -> dict:
    proc: subprocess.Popen = state["proc"]
    row = state["row"]
    elapsed = time.time() - float(state["started_at"])
    if timed_out:
        kill_process_tree(proc)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        kill_process_tree(proc)
        proc.wait()
    state["stdout_handle"].close()
    state["stderr_handle"].close()
    stdout_path: Path = state["stdout_path"]
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
    worker_report_path: Path = state["worker_report_path"]
    report_path: Path = state["report_path"]
    if timed_out:
        message = f"WorkerTimeoutExpired: {float(args.sample_timeout) + float(args.worker_timeout_margin):.1f}s"
        features = recover_timeout_features(worker_report_path, message, elapsed)
    else:
        features = parse_child_features(proc, stdout or "", row, elapsed)
    copy_report_if_available(worker_report_path, report_path)
    out = finalize_output(row, features, report_path, args, elapsed)
    if timed_out and not report_path.exists():
        out["extract_status"] = "worker_timeout"
    print(
        f"    status={out['extract_status']} success={out.get('speakeasy_success')} "
        f"timeout={out.get('error_timeout')} api={out.get('api_call_count')} "
        f"seq={out.get('sequence_event_count')} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return out


def stop_parallel_child(state: dict) -> None:
    proc: subprocess.Popen = state["proc"]
    if proc.poll() is None:
        kill_process_tree(proc)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    for handle_name in ("stdout_handle", "stderr_handle"):
        handle = state.get(handle_name)
        if handle and not handle.closed:
            handle.close()


def kill_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                text=True,
                capture_output=True,
                timeout=10,
            )
            return
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass


def run_pending_parallel(
    pending: list[tuple[dict, int]],
    total: int,
    total_pending_count: int,
    args: argparse.Namespace,
    features_path: Path,
    feature_rows: list[dict],
    fieldnames: list[str],
    start: float,
) -> int:
    reports_dir = args.output_dir / "reports"
    running: list[dict] = []
    completed_now = 0
    next_task = 0
    workers = max(1, int(args.workers or 1))
    worker_timeout = float(args.sample_timeout) + float(args.worker_timeout_margin)
    print(f"Running with {workers} Speakeasy child process(es) for {len(pending)} pending sample(s).", flush=True)
    try:
        while next_task < len(pending) or running:
            while next_task < len(pending) and len(running) < workers:
                row, index = pending[next_task]
                running.append(start_parallel_child(row, index, total, args, reports_dir))
                next_task += 1

            time.sleep(0.2)
            still_running: list[dict] = []
            for state in running:
                proc: subprocess.Popen = state["proc"]
                if proc.poll() is None:
                    if time.time() - float(state["started_at"]) > worker_timeout:
                        out = finish_parallel_child(state, args, reports_dir, timed_out=True)
                        append_csv(features_path, out, fieldnames)
                        feature_rows.append(out)
                        completed_now += 1
                        write_partial_summary(
                            args,
                            feature_rows,
                            total,
                            completed_now,
                            start,
                            max(0, total_pending_count - completed_now),
                        )
                        continue
                    still_running.append(state)
                    continue
                out = finish_parallel_child(state, args, reports_dir)
                append_csv(features_path, out, fieldnames)
                feature_rows.append(out)
                completed_now += 1
                write_partial_summary(
                    args,
                    feature_rows,
                    total,
                    completed_now,
                    start,
                    max(0, total_pending_count - completed_now),
                )
            running = still_running
    except BaseException:
        for state in running:
            stop_parallel_child(state)
        raise
    return completed_now


def summarize(rows: list[dict], args: argparse.Namespace) -> dict:
    by_status: dict[str, int] = {}
    by_label: dict[str, dict[str, int]] = {}
    for row in rows:
        status = row.get("extract_status") or "unknown"
        label = str(row.get("label") or "")
        by_status[status] = by_status.get(status, 0) + 1
        by_label.setdefault(label, {})
        by_label[label][status] = by_label[label].get(status, 0) + 1
    return {
        "protocol": "Offline Speakeasy-X sequence dataset extraction. Training should consume features_csv, not run Speakeasy inline.",
        "runtime_config": {
            "sample_timeout": args.sample_timeout,
            "emu_timeout": args.emu_timeout,
            "max_api_count": args.max_api_count,
            "max_instructions": args.max_instructions,
            "max_sequence_events": args.max_sequence_events,
            "disable_auto_mount": bool(args.disable_auto_mount),
            "workers": int(args.workers or 1),
            "worker_timeout_margin": args.worker_timeout_margin,
        },
        "rows": len(rows),
        "by_status": by_status,
        "by_label_status": by_label,
        "rows_with_api_sequence": sum(1 for row in rows if row.get("api_sequence")),
        "features_csv": str(args.output_dir / "features.csv"),
        "manifest_csv": str(args.output_dir / "manifest.csv"),
        "reports_dir": str(args.output_dir / "reports"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a resumable offline Speakeasy-X API sequence dataset.")
    parser.add_argument("--input-csv", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--speakeasy-root", type=Path, default=DEFAULT_SPEAKEASY_ROOT)
    parser.add_argument("--split", action="append", default=None, help="Optional split filter; can be repeated.")
    parser.add_argument("--label", type=int, choices=[0, 1], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--max-new-rows",
        type=int,
        default=None,
        help="Maximum newly extracted rows for this run after resume filtering; useful for bounded chunks.",
    )
    # With global wall-clock timeout in Speakeasy-X, emu_timeout is now the
    # TOTAL budget shared across all entry points and exports combined.
    # sample_timeout must be > emu_timeout to allow report writing and process exit.
    parser.add_argument("--sample-timeout", type=float, default=30.0)
    parser.add_argument("--emu-timeout", type=float, default=15.0)
    parser.add_argument("--max-api-count", type=int, default=1200)
    parser.add_argument("--max-instructions", type=int, default=500000)
    parser.add_argument("--max-sequence-events", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-timeout-margin", type=float, default=10.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--disable-auto-mount", dest="disable_auto_mount", action="store_true", default=True)
    parser.add_argument("--enable-auto-mount", dest="disable_auto_mount", action="store_false")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = args.input_csv or DEFAULT_INPUTS
    raw_rows: list[dict] = []
    for path in input_paths:
        raw_rows.extend(read_rows(path))

    selected = select_rows(raw_rows, args)
    manifest_path = args.output_dir / "manifest.csv"
    features_path = args.output_dir / "features.csv"
    if args.overwrite or not manifest_path.exists():
        write_csv(manifest_path, selected)

    existing = existing_rows(features_path)
    completed = {row.get("sample_id", "") for row in existing if completed_enough(row)}
    feature_rows = list(existing)
    fieldnames = feature_fieldnames(selected, existing)
    total = len(selected)
    completed_now = 0
    start = time.time()
    pending = [
        (row, index)
        for index, row in enumerate(selected, start=1)
        if not (args.resume and not args.overwrite and row["sample_id"] in completed)
    ]
    total_pending_count = len(pending)
    if args.max_new_rows is not None and args.max_new_rows > 0:
        pending = pending[: args.max_new_rows]
    write_partial_summary(args, feature_rows, total, completed_now, start, total_pending_count)
    workers = max(1, int(args.workers or 1))
    if workers == 1:
        for row, index in pending:
            out = extract_one(row, index, total, args)
            append_csv(features_path, out, fieldnames)
            feature_rows.append(out)
            completed_now += 1
            write_partial_summary(args, feature_rows, total, completed_now, start, max(0, total_pending_count - completed_now))
    else:
        completed_now = run_pending_parallel(
            pending,
            total,
            total_pending_count,
            args,
            features_path,
            feature_rows,
            fieldnames,
            start,
        )

    summary = summarize(feature_rows, args)
    summary["selected_rows"] = total
    summary["completed_rows"] = len(feature_rows)
    summary["pending_rows"] = max(0, total_pending_count - completed_now)
    summary["newly_extracted_rows"] = completed_now
    summary["elapsed_seconds"] = time.time() - start
    summary["complete"] = summary["pending_rows"] == 0
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    partial_summary_path = args.output_dir / "summary_partial.json"
    partial_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "features_csv": str(features_path),
        "selected_rows": total,
        "newly_extracted_rows": completed_now,
        "rows_with_api_sequence": summary["rows_with_api_sequence"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
