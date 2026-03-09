"""Build thermo anchor manifest JSON from truth-run records."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _as_float(value: Any, *, name: str) -> float:
    out = float(value)
    if not (out == out and abs(out) != float("inf")):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _extract_rpm_torque(rec: dict[str, Any]) -> tuple[float, float] | None:
    if "rpm" in rec and "torque" in rec:
        return _as_float(rec["rpm"], name="rpm"), _as_float(rec["torque"], name="torque")

    op = rec.get("operating_point")
    if isinstance(op, dict) and "rpm" in op and "torque" in op:
        return _as_float(op["rpm"], name="rpm"), _as_float(op["torque"], name="torque")

    cand = rec.get("candidate")
    if isinstance(cand, dict) and "rpm" in cand and "torque" in cand:
        return _as_float(cand["rpm"], name="rpm"), _as_float(cand["torque"], name="torque")

    return None


def _load_records(paths: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".jsonl":
            for lineno, raw in enumerate(text.splitlines(), start=1):
                line = raw.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    raise ValueError(f"{path}:{lineno}: expected JSON object")
                out.append(rec)
            continue
        payload = json.loads(text)
        if isinstance(payload, list):
            for i, rec in enumerate(payload):
                if not isinstance(rec, dict):
                    raise ValueError(f"{path}: list item {i} is not an object")
                out.append(rec)
        elif isinstance(payload, dict):
            out.append(payload)
        else:
            raise ValueError(f"{path}: expected JSON object or list")
    return out


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    inputs = [Path(p) for p in args.input]
    records = _load_records(inputs)

    seen: set[tuple[float, float]] = set()
    anchors: list[dict[str, Any]] = []
    for rec in records:
        if "truth_ok" in rec and not bool(rec.get("truth_ok", False)):
            continue
        pair = _extract_rpm_torque(rec)
        if pair is None:
            continue
        rpm, torque = pair
        if rpm <= 0.0 or torque < 0.0:
            continue
        key = (round(rpm, 6), round(torque, 6))
        if key in seen:
            continue
        seen.add(key)
        anchors.append(
            {
                "label": f"rpm_{int(round(rpm))}_torque_{int(round(torque))}",
                "rpm": float(rpm),
                "torque": float(torque),
                "source": str(args.source),
            }
        )

    anchors.sort(key=lambda x: (float(x["rpm"]), float(x["torque"])))
    if int(args.max_anchors) > 0:
        anchors = anchors[: int(args.max_anchors)]

    return {
        "version": str(args.version),
        "validated_envelope": {
            "rpm_min": float(args.rpm_min),
            "rpm_max": float(args.rpm_max),
            "torque_min": float(args.torque_min),
            "torque_max": float(args.torque_max),
        },
        "thresholds": {
            "delta_m_air_rel_max": float(args.delta_m_air_rel_max),
            "delta_residual_abs_max": float(args.delta_residual_abs_max),
            "delta_scavenging_abs_max": float(args.delta_scavenging_abs_max),
        },
        "provenance": {
            "generated_by": "tools/build_thermo_anchor_manifest.py",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "source_type": str(args.source),
            "input_files": [str(p) for p in inputs],
        },
        "anchors": anchors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build thermo anchor manifest from truth records")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input truth records (.json/.jsonl). Repeatable.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/thermo/anchor_manifest_v1.json",
        help="Output manifest path",
    )
    parser.add_argument("--version", type=str, default="thermo_anchor_v1")
    parser.add_argument("--source", type=str, default="truth_runs")
    parser.add_argument("--max-anchors", type=int, default=0, help="0 means keep all anchors")
    parser.add_argument("--rpm-min", type=float, default=1000.0)
    parser.add_argument("--rpm-max", type=float, default=7000.0)
    parser.add_argument("--torque-min", type=float, default=40.0)
    parser.add_argument("--torque-max", type=float, default=400.0)
    parser.add_argument("--delta-m-air-rel-max", type=float, default=0.10)
    parser.add_argument("--delta-residual-abs-max", type=float, default=0.05)
    parser.add_argument("--delta-scavenging-abs-max", type=float, default=0.08)
    args = parser.parse_args(argv)

    manifest = build_manifest(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote anchor manifest to {out} (anchors={len(manifest['anchors'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
