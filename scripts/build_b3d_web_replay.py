#!/usr/bin/env python3
"""Build a local browser replay artifact from an AddBiomechanics B3D trial."""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import nimblephysics as nimble
import numpy as np

from build_task1_force_sources import build_source, parse_mot
from extract_b3d_kinematics import pass_type_name, select_trial, trial_summary


DEFAULT_FLOOR_Y = -0.038
CONTACT_GATE_N = 800.0
CONTACT_MARGIN_M = 0.012


def parse_float_list(text: str | None) -> list[float]:
    return [float(value) for value in (text or "").split()]


def parse_int_list(text: str | None) -> list[int]:
    return [int(value) for value in (text or "").split()]


def rounded_vertices(vertices: np.ndarray) -> list[list[float]]:
    return np.round(vertices.astype(float), 6).tolist()


def triangulate_polygon(indices: list[int]) -> list[list[int]]:
    if len(indices) < 3:
        return []
    if len(indices) == 3:
        return [indices]
    return [[indices[0], indices[i], indices[i + 1]] for i in range(1, len(indices) - 1)]


def parse_vtp_mesh(path: Path, scale: np.ndarray) -> tuple[list[list[float]], list[list[int]]]:
    root = ET.parse(path).getroot()
    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError(f"{path} is missing VTP Piece")

    point_array = piece.find("./Points/DataArray")
    if point_array is None:
        raise ValueError(f"{path} is missing VTP points")
    point_values = parse_float_list(point_array.text)
    if len(point_values) % 3 != 0:
        raise ValueError(f"{path} point array length is not divisible by 3")
    vertices = np.asarray(point_values, dtype=np.float64).reshape((-1, 3)) * scale[None, :]

    polys = piece.find("./Polys")
    triangles: list[list[int]] = []
    if polys is not None:
        arrays = {
            data.attrib.get("Name", ""): data
            for data in polys.findall("./DataArray")
        }
        connectivity = parse_int_list(arrays.get("connectivity").text if arrays.get("connectivity") is not None else "")
        offsets = parse_int_list(arrays.get("offsets").text if arrays.get("offsets") is not None else "")
        start = 0
        for offset in offsets:
            face = connectivity[start:offset]
            triangles.extend(triangulate_polygon(face))
            start = offset

    return rounded_vertices(vertices), triangles


def find_geometry_file(mesh_file: str, geometry_dirs: list[Path]) -> Path | None:
    for geometry_dir in geometry_dirs:
        candidate = geometry_dir / mesh_file
        if candidate.exists():
            return candidate
    return None


def build_geometry_meshes(osim_text: str, body_names: list[str], geometry_dirs: list[Path]) -> list[dict[str, Any]]:
    if not geometry_dirs:
        return []

    root = ET.fromstring(osim_text)
    geometry_meshes: list[dict[str, Any]] = []
    missing: list[str] = []
    for body in root.findall(".//Body"):
        body_name = body.attrib.get("name", "")
        if body_name not in body_names:
            continue
        body_index = body_names.index(body_name)
        attached = body.find("./attached_geometry")
        if attached is None:
            continue
        for mesh in attached.findall("./Mesh"):
            mesh_file = (mesh.findtext("./mesh_file") or "").strip()
            if not mesh_file:
                continue
            mesh_path = find_geometry_file(mesh_file, geometry_dirs)
            if mesh_path is None:
                missing.append(mesh_file)
                continue
            scale_values = parse_float_list(mesh.findtext("./scale_factors"))
            scale = np.asarray(scale_values if len(scale_values) == 3 else [1.0, 1.0, 1.0], dtype=np.float64)
            vertices, triangles = parse_vtp_mesh(mesh_path, scale)
            geometry_meshes.append(
                {
                    "body": body_name,
                    "bodyIndex": body_index,
                    "meshName": mesh.attrib.get("name", ""),
                    "meshFile": mesh_file,
                    "vertices": vertices,
                    "triangles": triangles,
                }
            )

    if missing:
        unique_missing = ", ".join(sorted(set(missing)))
        print(f"Warning: missing geometry mesh files: {unique_missing}")
    return geometry_meshes


def normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-9:
        return fallback.astype(float)
    return vector / norm


def transform_sole(local: np.ndarray, calcn: np.ndarray, toes: np.ndarray) -> np.ndarray:
    world_up = np.array([0.0, 1.0, 0.0])
    x_axis = normalize(toes - calcn, np.array([1.0, 0.0, 0.0]))
    z_axis = normalize(np.cross(x_axis, world_up), np.array([0.0, 0.0, 1.0]))
    y_axis = normalize(np.cross(z_axis, x_axis), world_up)
    if float(np.dot(y_axis, world_up)) < 0:
        y_axis *= -1
        z_axis *= -1

    origin = 0.5 * (calcn + toes)
    return (
        origin[None, :]
        + local[:, 0:1] * x_axis[None, :]
        + local[:, 1:2] * y_axis[None, :]
        + local[:, 2:3] * z_axis[None, :]
    )


def load_template(template_dir: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    manifest = json.loads((template_dir / "manifest.json").read_text())
    right_local = np.fromfile(template_dir / manifest["buffers"]["rightSoleLocal"]["path"], dtype=np.float32)
    left_local = np.fromfile(template_dir / manifest["buffers"]["leftSoleLocal"]["path"], dtype=np.float32)
    node_count = int(manifest["sole"]["nodeCount"])
    return manifest, right_local.reshape(node_count, 3), left_local.reshape(node_count, 3)


def body_transform(skeleton: Any, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    transform = skeleton.getBodyNode(body_name).getWorldTransform()
    return np.asarray(transform.translation(), dtype=np.float64), np.asarray(transform.rotation(), dtype=np.float64)


def build_artifact(args: argparse.Namespace) -> None:
    template_manifest, right_local, left_local = load_template(args.template_artifact)
    subject = nimble.biomechanics.SubjectOnDisk(str(args.b3d))
    trial_index = select_trial(subject, args.trial_index, args.trial_name)
    summary = trial_summary(subject, trial_index)

    skeleton = subject.readSkel(args.processing_pass)
    body_names = [
        name
        for name in template_manifest["bodyNames"]
        if skeleton.getBodyNode(name) is not None
    ]
    missing_body_names = [name for name in template_manifest["bodyNames"] if name not in body_names]
    if missing_body_names:
        print(f"Warning: B3D skeleton is missing bodies: {missing_body_names}")

    geometry_meshes = build_geometry_meshes(
        subject.getOpensimFileText(args.processing_pass),
        body_names,
        args.geometry_dir,
    )

    frames = subject.readFrames(
        trial_index,
        0,
        summary["frames"],
        includeSensorData=False,
        includeProcessingPasses=True,
    )
    frame_count = len(frames)
    body_count = len(body_names)
    node_count = int(template_manifest["sole"]["nodeCount"])
    times = np.arange(frame_count, dtype=np.float64) * float(summary["timestepS"])

    body_trans = np.zeros((frame_count, body_count, 3), dtype=np.float32)
    body_rots = np.zeros((frame_count, body_count, 3, 3), dtype=np.float32)
    right_world = np.zeros((frame_count, node_count, 3), dtype=np.float32)
    left_world = np.zeros((frame_count, node_count, 3), dtype=np.float32)

    for frame_index, frame in enumerate(frames):
        skeleton.setPositions(frame.processingPasses[args.processing_pass].pos)
        for body_index, body_name in enumerate(body_names):
            translation, rotation = body_transform(skeleton, body_name)
            body_trans[frame_index, body_index] = translation
            body_rots[frame_index, body_index] = rotation

        calcn_r, _ = body_transform(skeleton, "calcn_r")
        toes_r, _ = body_transform(skeleton, "toes_r")
        calcn_l, _ = body_transform(skeleton, "calcn_l")
        toes_l, _ = body_transform(skeleton, "toes_l")
        right_world[frame_index] = transform_sole(right_local, calcn_r, toes_r)
        left_world[frame_index] = transform_sole(left_local, calcn_l, toes_l)

    bottom_ids = np.asarray(template_manifest["sole"]["bottomContactNodeIds"], dtype=np.int64)
    bottom_y = np.concatenate([
        right_world[:, bottom_ids, 1].min(axis=1),
        left_world[:, bottom_ids, 1].min(axis=1),
    ])
    stance_floor_estimate = float(np.percentile(bottom_y, args.floor_percentile))
    y_shift = args.floor_y - stance_floor_estimate
    body_trans[:, :, 1] += y_shift
    right_world[:, :, 1] += y_shift
    left_world[:, :, 1] += y_shift

    right_bottom = right_world[:, bottom_ids, 1].min(axis=1)
    left_bottom = left_world[:, bottom_ids, 1].min(axis=1)
    right_gate = np.where(right_bottom <= args.floor_y + CONTACT_MARGIN_M, CONTACT_GATE_N, 0.0)
    left_gate = np.where(left_bottom <= args.floor_y + CONTACT_MARGIN_M, CONTACT_GATE_N, 0.0)

    right_cop = 0.5 * (right_world[:, bottom_ids, :].min(axis=1) + right_world[:, bottom_ids, :].max(axis=1))
    left_cop = 0.5 * (left_world[:, bottom_ids, :].min(axis=1) + left_world[:, bottom_ids, :].max(axis=1))
    right_cop[:, 1] = args.floor_y
    left_cop[:, 1] = args.floor_y

    time_list = [round(float(value), 6) for value in times]
    gaitdynamics_source = None
    if args.gaitdynamics_grf:
        _, gaitdynamics_rows = parse_mot(args.gaitdynamics_grf)
        gaitdynamics_source = build_source(gaitdynamics_rows, [float(value) for value in times], "gaitdynamics")
        right_force = gaitdynamics_source["rightForceN"]
        left_force = gaitdynamics_source["leftForceN"]
        right_cop_series = gaitdynamics_source["rightCopM"]
        left_cop_series = gaitdynamics_source["leftCopM"]
    else:
        right_force = [[0.0, float(value), 0.0] for value in right_gate]
        left_force = [[0.0, float(value), 0.0] for value in left_gate]
        right_cop_series = right_cop.round(6).tolist()
        left_cop_series = left_cop.round(6).tolist()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    buffers = {
        "bodyTranslations": ("body_translations_f32.bin", body_trans),
        "bodyRotations": ("body_rotations_f32.bin", body_rots),
        "rightSoleWorld": ("right_sole_world_f32.bin", right_world),
        "leftSoleWorld": ("left_sole_world_f32.bin", left_world),
        "rightSoleLocal": ("right_sole_local_f32.bin", right_local.astype(np.float32)),
        "leftSoleLocal": ("left_sole_local_f32.bin", left_local.astype(np.float32)),
    }
    buffer_manifest = {}
    for key, (filename, array) in buffers.items():
        array.astype(np.float32).tofile(out_dir / filename)
        buffer_manifest[key] = {
            "path": filename,
            "shape": list(array.shape),
            "dtype": "float32",
        }

    for filename in ["sole_floor_contact_metadata.json", "sole_floor_contact.onnx"]:
        shutil.copy2(args.template_artifact / filename, out_dir / filename)

    all_points = np.concatenate([
        body_trans.reshape(-1, 3),
        right_world.reshape(-1, 3),
        left_world.reshape(-1, 3),
    ], axis=0)
    pad = np.array([0.4, 0.2, 0.4])
    bounds_min = (all_points.min(axis=0) - pad).round(6).tolist()
    bounds_max = (all_points.max(axis=0) + pad).round(6).tolist()

    manifest = dict(template_manifest)
    zero3 = [[0.0, 0.0, 0.0] for _ in range(frame_count)]
    manifest.update(
        {
            "schema": "gait_nnfe_web_replay.v1",
            "frames": frame_count,
            "durationS": round(float(times[-1] if frame_count else 0.0), 6),
            "bodyNames": body_names,
            "geometryMeshes": geometry_meshes,
            "bounds": {"min": bounds_min, "max": bounds_max},
            "buffers": buffer_manifest,
            "timeseries": {
                "time": time_list,
                "rightForceN": right_force,
                "leftForceN": left_force,
                "rightProjectedCopM": right_cop_series,
                "leftProjectedCopM": left_cop_series,
                "rightProjectedCopLocalM": zero3,
                "leftProjectedCopLocalM": zero3,
                "rightCopWasClipped": [False for _ in range(frame_count)],
                "leftCopWasClipped": [False for _ in range(frame_count)],
                "rightMaxCompressionMm": [0.0 for _ in range(frame_count)],
                "leftMaxCompressionMm": [0.0 for _ in range(frame_count)],
                "rightBottomOutMarginMm": [
                    round(float(template_manifest["sole"]["thicknessM"]) * 1000.0, 6)
                    for _ in range(frame_count)
                ],
                "leftBottomOutMarginMm": [
                    round(float(template_manifest["sole"]["thicknessM"]) * 1000.0, 6)
                    for _ in range(frame_count)
                ],
            },
            "defaults": {
                **template_manifest.get("defaults", {}),
                "floorY": args.floor_y,
                "deformationExaggeration": 1,
                "geometryOpacity": 0.5,
                "pressureColorMaxMpa": 0.6,
            },
            "provenance": {
                "source": "AddBiomechanics B3D",
                "sourceB3d": args.b3d.name,
                "trial": summary,
                "processingPassIndex": args.processing_pass,
                "processingPassType": pass_type_name(subject, args.processing_pass),
                "floorY": args.floor_y,
                "floorPercentile": args.floor_percentile,
                "gaitDynamicsOutput": args.gaitdynamics_grf.name if args.gaitdynamics_grf else None,
                "geometryMeshCount": len(geometry_meshes),
                "note": (
                    "Local AddBiomechanics artifact using GaitDynamics-predicted GRF/CoP."
                    if args.gaitdynamics_grf
                    else "Local preview artifact. Force series is a contact gate until GaitDynamics prediction is wired in."
                ),
            },
        }
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    force_source = gaitdynamics_source or {
        "rightForceN": right_force,
        "leftForceN": left_force,
        "rightCopM": right_cop_series,
        "leftCopM": left_cop_series,
    }
    (out_dir / "task1_force_sources.json").write_text(
        json.dumps(
            {
                "schema": "gait_nnfe_task1_force_sources.v1",
                "description": (
                    "GaitDynamics-predicted GRF/CoP from AddBiomechanics kinematics."
                    if args.gaitdynamics_grf
                    else "Local AddBiomechanics preview force source. This is a contact-gate placeholder."
                ),
                "defaultSourceId": "gaitDynamicsPredicted",
                "time": time_list,
                "provenance": {
                    "task": "AddBiomechanics kinematics-only local preview",
                    "kinematicTrial": summary["trialName"],
                    "sourceB3d": args.b3d.name,
                    "processingPassIndex": args.processing_pass,
                    "processingPassType": pass_type_name(subject, args.processing_pass),
                    "gaitDynamicsOutput": args.gaitdynamics_grf.name if args.gaitdynamics_grf else None,
                    "fileMapping": {
                        "b3dInput": args.b3d.name,
                        "extractedKinematicsMot": "generated separately by extract_b3d_kinematics.py",
                        "gaitDynamicsPredictedForceMot": args.gaitdynamics_grf.name if args.gaitdynamics_grf else None,
                    },
                },
                "sources": {
                    "gaitDynamicsPredicted": {
                        "label": "GaitDynamics prediction" if args.gaitdynamics_grf else "Contact-gated preview",
                        "kind": "prediction" if args.gaitdynamics_grf else "placeholder",
                        "lineStyle": "dashed",
                        **force_source,
                    }
                },
                "metrics": {},
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Wrote local web artifact: {out_dir}")
    print(f"  frames: {frame_count}")
    print(f"  trial: {summary['trialName']}")
    print(f"  processing pass: {args.processing_pass} {pass_type_name(subject, args.processing_pass)}")
    print(f"  y shift: {y_shift:.4f} m")
    if args.gaitdynamics_grf:
        print(f"  gaitdynamics grf: {args.gaitdynamics_grf}")
    else:
        print(f"  right gated frames: {int(np.count_nonzero(right_gate))}")
        print(f"  left gated frames: {int(np.count_nonzero(left_gate))}")
    print(f"  geometry meshes: {len(geometry_meshes)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b3d", type=Path, required=True)
    parser.add_argument("--trial-index", type=int)
    parser.add_argument("--trial-name")
    parser.add_argument("--processing-pass", type=int, default=0)
    parser.add_argument(
        "--template-artifact",
        type=Path,
        default=Path("web/public/artifacts/gait_nnfe_replay"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output browser replay artifact directory, for example web/public/artifacts/addb_subject2_walkingTS1_replay.",
    )
    parser.add_argument("--floor-y", type=float, default=DEFAULT_FLOOR_Y)
    parser.add_argument("--floor-percentile", type=float, default=12.0)
    parser.add_argument(
        "--gaitdynamics-grf",
        type=Path,
        help="Optional GaitDynamics *_grf_pred___.mot file to use as the aggregate GRF/CoP source.",
    )
    parser.add_argument(
        "--geometry-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory containing OpenSim .vtp mesh files referenced by the B3D/OpenSim model. Repeatable.",
    )
    args = parser.parse_args()
    build_artifact(args)


if __name__ == "__main__":
    main()
