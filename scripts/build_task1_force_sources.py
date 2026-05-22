#!/usr/bin/env python3
"""Build browser force-source artifacts for GaitDynamics Task 1 validation."""

from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path


FORCE_THRESHOLD_N = 25.0


def parse_mot(path: Path) -> tuple[list[str], list[dict[str, float]]]:
    lines = path.read_text().splitlines()
    header_index = None
    for index, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            header_index = index + 1
            break
    if header_index is None:
        raise ValueError(f"{path} is missing endheader")

    columns = lines[header_index].strip().split()
    rows = []
    for line in lines[header_index + 1 :]:
        if not line.strip():
            continue
        values = [float(value) for value in line.strip().split()]
        if len(values) != len(columns):
            raise ValueError(f"{path} row has {len(values)} values; expected {len(columns)}")
        rows.append(dict(zip(columns, values)))
    return columns, rows


def resample_series(rows: list[dict[str, float]], target_times: list[float], columns: list[str]) -> list[list[float]]:
    source_times = [row["time"] for row in rows]
    series = []
    for time in target_times:
        if time <= source_times[0]:
            series.append([rows[0][column] for column in columns])
            continue
        if time >= source_times[-1]:
            series.append([rows[-1][column] for column in columns])
            continue
        right = bisect.bisect_left(source_times, time)
        left = right - 1
        t0 = source_times[left]
        t1 = source_times[right]
        alpha = 0.0 if t1 == t0 else (time - t0) / (t1 - t0)
        series.append([
            rows[left][column] + alpha * (rows[right][column] - rows[left][column])
            for column in columns
        ])
    return series


def build_source(rows: list[dict[str, float]], target_times: list[float], prefix: str) -> dict[str, list[list[float]]]:
    if prefix == "opensim":
        right_force_cols = ["ground_force_r_vx", "ground_force_r_vy", "ground_force_r_vz"]
        left_force_cols = ["ground_force_l_vx", "ground_force_l_vy", "ground_force_l_vz"]
        right_cop_cols = ["ground_force_r_px", "ground_force_r_py", "ground_force_r_pz"]
        left_cop_cols = ["ground_force_l_px", "ground_force_l_py", "ground_force_l_pz"]
    elif prefix == "gaitdynamics":
        right_force_cols = ["force1_vx", "force1_vy", "force1_vz"]
        left_force_cols = ["force2_vx", "force2_vy", "force2_vz"]
        right_cop_cols = ["force1_px", "force1_py", "force1_pz"]
        left_cop_cols = ["force2_px", "force2_py", "force2_pz"]
    else:
        raise ValueError(f"Unknown source prefix {prefix}")

    return {
        "rightForceN": resample_series(rows, target_times, right_force_cols),
        "leftForceN": resample_series(rows, target_times, left_force_cols),
        "rightCopM": resample_series(rows, target_times, right_cop_cols),
        "leftCopM": resample_series(rows, target_times, left_cop_cols),
    }


def vector_norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def rmse(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / max(len(values), 1))


def side_metrics(reference: dict[str, list[list[float]]], predicted: dict[str, list[list[float]]], side: str) -> dict[str, float]:
    force_key = f"{side}ForceN"
    cop_key = f"{side}CopM"
    ref_vertical = [force[1] for force in reference[force_key]]
    pred_vertical = [force[1] for force in predicted[force_key]]
    vertical_errors = [pred - ref for ref, pred in zip(ref_vertical, pred_vertical)]

    ref_force_norm = [vector_norm(force) for force in reference[force_key]]
    pred_force_norm = [vector_norm(force) for force in predicted[force_key]]
    cop_errors = []
    for ref_norm, pred_norm, ref_cop, pred_cop in zip(
        ref_force_norm,
        pred_force_norm,
        reference[cop_key],
        predicted[cop_key],
    ):
        if ref_norm < FORCE_THRESHOLD_N or pred_norm < FORCE_THRESHOLD_N:
            continue
        cop_errors.append(vector_norm([pred_cop[i] - ref_cop[i] for i in range(3)]))

    peak_ref = max(ref_vertical) if ref_vertical else 0.0
    peak_pred = max(pred_vertical) if pred_vertical else 0.0
    stance_ref = sum(value >= FORCE_THRESHOLD_N for value in ref_vertical)
    stance_pred = sum(value >= FORCE_THRESHOLD_N for value in pred_vertical)
    return {
        "verticalRmseN": rmse(vertical_errors),
        "verticalMaeN": sum(abs(value) for value in vertical_errors) / max(len(vertical_errors), 1),
        "peakReferenceN": peak_ref,
        "peakPredictedN": peak_pred,
        "peakErrorN": peak_pred - peak_ref,
        "copRmseM": rmse(cop_errors) if cop_errors else None,
        "stanceFrameDifference": stance_pred - stance_ref,
    }


def rounded(obj):
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, list):
        return [rounded(value) for value in obj]
    if isinstance(obj, dict):
        return {key: rounded(value) for key, value in obj.items()}
    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-grf", type=Path, required=True)
    parser.add_argument("--gaitdynamics-grf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    target_times = [float(value) for value in manifest["timeseries"]["time"]]
    _, reference_rows = parse_mot(args.reference_grf)
    _, gaitdynamics_rows = parse_mot(args.gaitdynamics_grf)

    reference = build_source(reference_rows, target_times, "opensim")
    gaitdynamics = build_source(gaitdynamics_rows, target_times, "gaitdynamics")
    metrics = {
        "right": side_metrics(reference, gaitdynamics, "right"),
        "left": side_metrics(reference, gaitdynamics, "left"),
    }
    all_vertical_errors = []
    for side in ("right", "left"):
        force_key = f"{side}ForceN"
        all_vertical_errors.extend(
            pred[1] - ref[1]
            for ref, pred in zip(reference[force_key], gaitdynamics[force_key])
        )
    metrics["combinedVerticalRmseN"] = rmse(all_vertical_errors)

    output = {
        "schema": "gait_nnfe_task1_force_sources.v1",
        "description": "Task 1 validation force sources: OpenSim reference GRF and GaitDynamics-predicted GRF from kinematics.",
        "defaultSourceId": "referenceOpenSim",
        "time": target_times,
        "provenance": {
            "task": "GaitDynamics Task 1 force estimation from kinematics",
            "kinematicTrial": "OpenSim 4.5 Moco exampleEMGTracking coordinates.mot",
            "referenceForce": "OpenSim 4.5 Moco exampleEMGTracking grf.mot",
            "gaitDynamicsInput": "coordinates.mot + subject_walk_armless_18musc.osim",
            "gaitDynamicsOutput": "coordinates_grf_pred___.mot",
            "gaitDynamicsCheckpoint": "GaitDynamics/example_usage/GaitDynamicsRefinement.pt",
            "fileMapping": {
                "kinematicInputMot": "coordinates.mot",
                "modelInputOsim": "subject_walk_armless_18musc.osim",
                "referenceForceMot": "grf.mot",
                "gaitDynamicsPredictedForceMot": "coordinates_grf_pred___.mot",
            },
        },
        "sources": {
            "referenceOpenSim": {
                "label": "OpenSim reference",
                "kind": "reference",
                "lineStyle": "solid",
                **reference,
            },
            "gaitDynamicsPredicted": {
                "label": "GaitDynamics prediction",
                "kind": "prediction",
                "lineStyle": "dashed",
                **gaitdynamics,
            },
        },
        "metrics": metrics,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rounded(output), indent=2) + "\n")


if __name__ == "__main__":
    main()
