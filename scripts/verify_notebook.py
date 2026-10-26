#!/usr/bin/env python3
"""Execute the complete notebook and validate its saved outputs."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from collections import Counter


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "AA_Assignment.ipynb"
CATEGORIES = {
    "business": 510,
    "entertainment": 386,
    "politics": 417,
    "sport": 511,
    "tech": 401,
}
EXPORT_COLUMNS = ["category", "text", "cleansed_text", "keywords"]
SCORE_LABELS = [
    "logistic_regression_word_counts",
    "logistic_regression_tfidf",
    "decision_tree_word_counts",
    "decision_tree_tfidf",
    "random_forest_word_counts",
    "random_forest_tfidf",
]


class VerificationError(Exception):
    """A notebook result failed a required check."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(
            reader.fieldnames == expected_columns,
            f"{path.name}: expected columns {expected_columns}, got {reader.fieldnames}",
        )
        rows = list(reader)
    require(len(rows) == 2225, f"{path.name}: expected 2,225 rows, got {len(rows):,}")
    for number, row in enumerate(rows, start=2):
        require(
            set(row) == set(expected_columns)
            and all(isinstance(value, str) and value.strip() for value in row.values()),
            f"{path.name}: missing or malformed data at CSV row {number}",
        )
    require(
        Counter(row["category"] for row in rows) == CATEGORIES,
        f"{path.name}: category counts differ from the original dataset",
    )
    return rows


def validate_export(source_rows: list[dict[str, str]]) -> dict:
    path = ROOT / "bbc-text_export.csv"
    require(path.is_file(), "Notebook did not create bbc-text_export.csv")
    rows = read_csv(path, EXPORT_COLUMNS)
    for number, (source, row) in enumerate(zip(source_rows, rows), start=2):
        require(
            row["category"] == source["category"] and row["text"] == source["text"],
            f"bbc-text_export.csv: source article or order changed at CSV row {number}",
        )
        try:
            keywords = ast.literal_eval(row["keywords"])
        except (ValueError, SyntaxError) as exc:
            raise VerificationError(
                f"bbc-text_export.csv: invalid keyword dictionary at CSV row {number}"
            ) from exc
        require(
            isinstance(keywords, dict) and 1 <= len(keywords) <= 5,
            f"bbc-text_export.csv: expected 1–5 keywords at CSV row {number}",
        )
        require(
            all(
                isinstance(word, str)
                and word.strip()
                and isinstance(weight, (int, float))
                and not isinstance(weight, bool)
                and math.isfinite(weight)
                and 0 <= weight <= 1
                for word, weight in keywords.items()
            ),
            f"bbc-text_export.csv: invalid keyword or TF-IDF weight at CSV row {number}",
        )
    return {"rows": len(rows), "columns": EXPORT_COLUMNS, "sha256": sha256(path)}


def test_scores(notebook) -> dict[str, dict]:
    scores = []
    for index, cell in enumerate(notebook.cells):
        for output in cell.get("outputs", []):
            text = output.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            for value in re.findall(r"(?m)^Test score:\s*([^\s]+)\s*$", text):
                try:
                    accuracy = float(value)
                except ValueError as exc:
                    raise VerificationError(f"Invalid test score in cell {index + 1}") from exc
                require(
                    math.isfinite(accuracy) and 0 <= accuracy <= 1,
                    f"Test score outside [0, 1] in cell {index + 1}",
                )
                scores.append({"cell": index + 1, "accuracy": accuracy})
    require(len(scores) == 6, f"Expected six test scores, found {len(scores)}")
    return dict(zip(SCORE_LABELS, scores))


def versions() -> dict[str, str]:
    result = {"python": sys.version.split()[0]}
    for package in (
        "nbclient", "nbformat", "ipykernel", "numpy", "pandas", "scipy",
        "scikit-learn", "matplotlib", "wordcloud", "mlxtend", "nltk",
    ):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not installed"
    return result


def positive_timeout(value: str) -> int:
    timeout = int(value)
    if timeout <= 0:
        raise argparse.ArgumentTypeError("timeout must be a positive number of seconds")
    return timeout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout", type=positive_timeout, default=1800,
        help="maximum execution time per cell in seconds (default: 1800)",
    )
    args = parser.parse_args()
    for name in (NOTEBOOK.name, "bbc-text.csv", "stopwords.txt"):
        if not (ROOT / name).is_file():
            print(f"Verification failed: required file is missing: {ROOT / name}", file=sys.stderr)
            return 1

    run_dir = ROOT / ".verification"
    # Set these before importing Jupyter or Matplotlib so their caches stay local.
    for variable, directory in {
        "JUPYTER_CONFIG_DIR": "jupyter/config",
        "JUPYTER_DATA_DIR": "jupyter/data",
        "JUPYTER_RUNTIME_DIR": "jupyter/runtime",
        "IPYTHONDIR": "ipython",
        "MPLCONFIGDIR": "matplotlib",
        "XDG_CACHE_HOME": "cache",
        "PYTHONPYCACHEPREFIX": "pycache",
    }.items():
        path = run_dir / directory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"

    started = time.monotonic()
    inputs = {name: sha256(ROOT / name) for name in ("bbc-text.csv", "stopwords.txt")}
    notebook_hash = sha256(NOTEBOOK)
    report = {"status": "failed", "versions": versions(), "input_sha256": inputs}
    notebook = None
    client = None
    manager = None
    completed = set()
    exit_code = 1
    try:
        import nbformat
        from nbclient import NotebookClient
        from jupyter_client import KernelManager

        source_rows = read_csv(ROOT / "bbc-text.csv", ["category", "text"])
        stopwords = (ROOT / "stopwords.txt").read_text(encoding="utf-8").splitlines()
        require(stopwords and all(word.strip() for word in stopwords), "stopwords.txt is empty or contains blank entries")
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        code_indices = [
            index for index, cell in enumerate(notebook.cells)
            if cell.cell_type == "code" and cell.source.strip()
        ]
        require(bool(code_indices), "Notebook contains no executable code cells")
        ordinals = {index: number for number, index in enumerate(code_indices, start=1)}
        cell_starts = {}
        report["code_cell_count"] = len(code_indices)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
                cell.metadata.pop("execution", None)

        def on_cell_start(cell, cell_index):
            if cell_index in ordinals:
                cell_starts[cell_index] = time.monotonic()
                print(f"[{ordinals[cell_index]}/{len(code_indices)}] Cell {cell_index + 1}: start", flush=True)

        def on_cell_executed(cell, cell_index, execute_reply):
            if cell_index in ordinals:
                elapsed = time.monotonic() - cell_starts[cell_index]
                status = execute_reply.get("content", {}).get("status", "unknown")
                if status == "ok":
                    completed.add(cell_index)
                print(
                    f"[{ordinals[cell_index]}/{len(code_indices)}] Cell {cell_index + 1}: {status} ({elapsed:.1f}s)",
                    flush=True,
                )

        manager = KernelManager(
            kernel_name="python3",
            connection_file=str(run_dir / "jupyter/runtime/kernel.json"),
        )
        # Do not resolve this path: resolving a venv symlink can select its base Python.
        manager.kernel_spec.argv = [
            os.path.abspath(sys.executable), "-m", "ipykernel_launcher", "-f", "{connection_file}",
        ]
        client = NotebookClient(
            notebook, km=manager, timeout=args.timeout,
            allow_errors=False, force_raise_errors=True, skip_cells_with_tag="",
            record_timing=True, on_cell_start=on_cell_start, on_cell_executed=on_cell_executed,
        )
        print(f"Executing {len(code_indices)} code cells with {sys.executable}", flush=True)
        client.execute(cwd=str(ROOT), env=os.environ.copy())
        require(completed == set(code_indices), "Not every nonblank code cell completed successfully")
        require(
            all(notebook.cells[index].execution_count is not None for index in code_indices),
            "A nonblank code cell has no execution count",
        )
        require(
            not any(output.get("output_type") == "error" for cell in notebook.cells for output in cell.get("outputs", [])),
            "Notebook contains error outputs",
        )
        report["test_scores"] = test_scores(notebook)
        report["export"] = validate_export(source_rows)
        report["figure_count"] = sum(
            "image/png" in output.get("data", {})
            for cell in notebook.cells for output in cell.get("outputs", [])
        )
        require(report["figure_count"] >= 20, f"Expected at least 20 figures, found {report['figure_count']}")
        require(all(sha256(ROOT / name) == digest for name, digest in inputs.items()), "A source input changed during execution")
        require(sha256(NOTEBOOK) == notebook_hash, "Notebook was edited during verification; fresh outputs were not saved")
        pending = run_dir / "executed-notebook.ipynb"
        nbformat.write(notebook, pending)
        os.replace(pending, NOTEBOOK)
        report["status"] = "passed"
        exit_code = 0
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"Verification failed: {report['error']}", file=sys.stderr, flush=True)
        if notebook is not None:
            import nbformat
            nbformat.write(notebook, run_dir / "failed-notebook.ipynb")
        if isinstance(exc, KeyboardInterrupt):
            exit_code = 130
    finally:
        if manager is not None and manager.has_kernel:
            manager.shutdown_kernel(now=True)
        if client is not None and client.kc is not None:
            client.kc.stop_channels()
        report["executed_code_cell_count"] = len(completed)
        report["inputs_unchanged"] = all(
            (ROOT / name).is_file() and sha256(ROOT / name) == digest
            for name, digest in inputs.items()
        )
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if exit_code == 0:
        print(f"Verification passed in {report['elapsed_seconds']:.1f}s. Saved fresh notebook outputs and .verification/report.json", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
