"""Collate cross-validation results into a single comparison table.

Scans an outputs directory for ``cv_results.json`` files (written by
``src.cross_validate``) and produces one mean +/- std comparison table, ready
to drop into a report. Output is printed to the console and saved as both CSV
and Markdown.

Usage:
    python -m src.collate_cv
    python -m src.collate_cv --results-dir outputs --output-dir outputs
    python -m src.collate_cv --sort accuracy
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = ["accuracy", "macro_f1", "weighted_f1"]


def _load_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load every cv_results.json found under *results_dir* (recursively)."""
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.rglob("cv_results.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [skip] {path}: {exc}")
            continue

        summary = data.get("summary", {})
        if not all(m in summary for m in METRICS):
            print(f"  [skip] {path}: missing summary metrics")
            continue

        row: dict[str, Any] = {
            "experiment": data.get("experiment_name") or path.parent.name,
            "mode": data.get("mode", "?"),
            "teacher": data.get("teacher_model_name") or "-",
            "student/model": data.get("model_name") or "-",
            "n_folds": data.get("n_folds"),
            "data_dir": data.get("data_dir", ""),
            "params": data.get("parameters"),
            "latency_ms": data.get("latency_ms"),
        }
        for m in METRICS:
            row[f"{m}_mean"] = summary[m]["mean"]
            row[f"{m}_std"] = summary[m]["std"]
        rows.append(row)
    return rows


def _fmt_pct(mean: float, std: float) -> str:
    return f"{mean * 100:.2f} +/- {std * 100:.2f}"


def _fmt_pct_md(mean: float, std: float) -> str:
    return f"{mean * 100:.2f} ± {std * 100:.2f}"


def _print_console(rows: list[dict[str, Any]]) -> None:
    headers = ["Experiment", "Teacher", "Student", "Params", "Acc %", "MacroF1 %", "WtF1 %", "Lat ms"]
    table = []
    for r in rows:
        table.append([
            r["experiment"],
            r["teacher"],
            r["student/model"],
            f"{r['params']:,}" if r["params"] is not None else "-",
            _fmt_pct(r["accuracy_mean"], r["accuracy_std"]),
            _fmt_pct(r["macro_f1_mean"], r["macro_f1_std"]),
            _fmt_pct(r["weighted_f1_mean"], r["weighted_f1_std"]),
            f"{r['latency_ms']:.2f}" if r["latency_ms"] is not None else "-",
        ])

    widths = [max(len(str(h)), *(len(str(row[i])) for row in table)) for i, h in enumerate(headers)]
    sep = "  "
    print(sep.join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print(sep.join("-" * widths[i] for i in range(len(headers))))
    for row in table:
        print(sep.join(str(c).ljust(widths[i]) for i, c in enumerate(row)))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "experiment", "mode", "teacher", "student/model", "n_folds", "data_dir",
        "params", "latency_ms",
        *(f"{m}_{stat}" for m in METRICS for stat in ("mean", "std")),
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    headers = ["Experiment", "Teacher", "Student", "Params", "Accuracy %", "Macro-F1 %", "Weighted-F1 %", "Latency (ms)"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join([
            r["experiment"],
            r["teacher"],
            r["student/model"],
            f"{r['params']:,}" if r["params"] is not None else "-",
            _fmt_pct_md(r["accuracy_mean"], r["accuracy_std"]),
            _fmt_pct_md(r["macro_f1_mean"], r["macro_f1_std"]),
            _fmt_pct_md(r["weighted_f1_mean"], r["weighted_f1_std"]),
            f"{r['latency_ms']:.2f}" if r["latency_ms"] is not None else "-",
        ]) + " |")
    n = rows[0]["n_folds"] if rows else "?"
    caption = f"\n_Cross-validated results ({n}-fold), mean ± standard deviation across folds._\n"
    path.write_text("\n".join(lines) + "\n" + caption, encoding="utf-8")


def _tex_escape(text: str) -> str:
    """Escape the LaTeX special characters that appear in model/teacher names."""
    return str(text).replace("_", r"\_")


def _write_latex(rows: list[dict[str, Any]], path: Path) -> None:
    """Write a standalone booktabs table for \\input into the LNCS report."""
    n = rows[0]["n_folds"] if rows else "?"
    data_dir = next((r["data_dir"] for r in rows if r.get("data_dir")), "")
    dataset = Path(data_dir).name if data_dir else "the target dataset"

    body = []
    for r in rows:
        teacher = "--" if r["teacher"] == "-" else _tex_escape(r["teacher"])
        params = f"{r['params']:,}" if r["params"] is not None else "--"
        body.append(
            f"    {teacher} & {_tex_escape(r['student/model'])} & "
            f"{_fmt_pct_md(r['accuracy_mean'], r['accuracy_std']).replace(chr(177), r'$\pm$')} & "
            f"{_fmt_pct_md(r['macro_f1_mean'], r['macro_f1_std']).replace(chr(177), r'$\pm$')} & "
            f"{_fmt_pct_md(r['weighted_f1_mean'], r['weighted_f1_std']).replace(chr(177), r'$\pm$')} & "
            f"{params} \\\\"
        )

    tex = (
        "% Auto-generated by src.collate_cv -- do not edit by hand.\n"
        "\\begin{table}\n"
        "  \\centering\n"
        f"  \\caption{{Cross-validated test results on {_tex_escape(dataset)} "
        f"({n}-fold, mean $\\pm$ standard deviation across folds). The teacher is "
        "retrained per fold; the student and KD hyperparameters are fixed.}\n"
        "  \\label{tab:cv-results}\n"
        "  \\resizebox{\\textwidth}{!}{\n"
        "  \\begin{tabular}{llrrrr}\n"
        "    \\toprule\n"
        "    Teacher & Student & Accuracy (\\%) & Macro-F1 (\\%) & Weighted-F1 (\\%) & Parameters \\\\\n"
        "    \\midrule\n"
        + "\n".join(body) + "\n"
        "    \\bottomrule\n"
        "  \\end{tabular}}\n"
        "\\end{table}\n"
    )
    path.write_text(tex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collate cross-validation results into one table.")
    parser.add_argument("--results-dir", default="outputs", help="Directory to scan for cv_results.json.")
    parser.add_argument("--output-dir", default="outputs", help="Where to write cv_summary.{csv,md}.")
    parser.add_argument(
        "--sort", default="accuracy", choices=[*METRICS, "experiment", "params"],
        help="Sort key (metrics/params sort descending; experiment ascending).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    rows = _load_results(results_dir)
    if not rows:
        print(f"No cv_results.json files found under {results_dir.resolve()}.")
        print("Run experiments first, e.g.:")
        print("  python -m src.cross_validate --config configs/cv_distillation_resnet50_cassava.yaml")
        return

    if args.sort == "experiment":
        rows.sort(key=lambda r: r["experiment"])
    elif args.sort == "params":
        rows.sort(key=lambda r: r["params"] or 0, reverse=True)
    else:
        rows.sort(key=lambda r: r[f"{args.sort}_mean"], reverse=True)

    print(f"\nFound {len(rows)} cross-validation result(s) under {results_dir.resolve()}\n")
    _print_console(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cv_summary.csv"
    md_path = output_dir / "cv_summary.md"
    tex_path = output_dir / "cv_summary.tex"
    _write_csv(rows, csv_path)
    _write_markdown(rows, md_path)
    _write_latex(rows, tex_path)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {tex_path}")


if __name__ == "__main__":
    main()
