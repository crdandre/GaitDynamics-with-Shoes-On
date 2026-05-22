#!/usr/bin/env python3
"""Extract a kinematics-only AddBiomechanics B3D trial to OpenSim files.

GaitDynamics has two practical input modes:

- Its dataset/training code reads AddBiomechanics `.b3d` files directly through
  Nimble `SubjectOnDisk`.
- Its public/example inference path consumes a subject `.osim` plus kinematic
  `.mot` files.

This script creates that second, portable representation from a single B3D
trial while using the kinematics processing pass by default.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import nimblephysics as nimble


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_") or "trial"


def pass_type_name(subject: Any, pass_index: int) -> str:
    try:
        return str(subject.getProcessingPassType(pass_index)).split(".")[-1]
    except Exception:
        return f"pass_{pass_index}"


def dof_names(subject: Any, processing_pass: int) -> list[str]:
    skeleton = subject.readSkel(processing_pass)
    return [skeleton.getDofByIndex(index).getName() for index in range(skeleton.getNumDofs())]


def trial_summary(subject: Any, trial_index: int) -> dict[str, Any]:
    missing = subject.getMissingGRF(trial_index)
    missing_count = sum(
        reason != nimble.biomechanics.MissingGRFReason.notMissingGRF
        for reason in missing
    )
    return {
        "trialIndex": trial_index,
        "trialName": subject.getTrialName(trial_index),
        "originalName": subject.getTrialOriginalName(trial_index),
        "frames": subject.getTrialLength(trial_index),
        "timestepS": subject.getTrialTimestep(trial_index),
        "durationS": subject.getTrialLength(trial_index) * subject.getTrialTimestep(trial_index),
        "processingPasses": subject.getTrialNumProcessingPasses(trial_index),
        "missingGrfFrames": missing_count,
        "tags": list(subject.getTrialTags(trial_index)),
    }


def select_trial(subject: Any, trial_index: int | None, trial_name: str | None) -> int:
    if trial_index is not None and trial_name is not None:
        raise ValueError("Use either --trial-index or --trial-name, not both")
    if trial_index is not None:
        if not 0 <= trial_index < subject.getNumTrials():
            raise ValueError(f"Trial index {trial_index} is out of range")
        return trial_index
    if trial_name is not None:
        matches = [
            index
            for index in range(subject.getNumTrials())
            if subject.getTrialName(index) == trial_name
            or subject.getTrialOriginalName(index) == trial_name
        ]
        if not matches:
            raise ValueError(f"No trial named {trial_name!r}")
        if len(matches) > 1:
            raise ValueError(f"Trial name {trial_name!r} matched multiple trials: {matches}")
        return matches[0]

    walking = [
        index
        for index in range(subject.getNumTrials())
        if "walk" in subject.getTrialName(index).lower()
        or "walk" in subject.getTrialOriginalName(index).lower()
    ]
    candidates = walking or list(range(subject.getNumTrials()))
    return max(candidates, key=lambda index: subject.getTrialLength(index))


def write_mot(
    path: Path,
    columns: list[str],
    poses: list[list[float]],
    timestep_s: float,
) -> None:
    with path.open("w") as out_file:
        out_file.write("Coordinates\n")
        out_file.write("version=1\n")
        out_file.write(f"nRows={len(poses)}\n")
        out_file.write(f"nColumns={len(columns) + 1}\n")
        out_file.write("inDegrees=no\n\n")
        out_file.write(
            "If the header above contains a line with 'inDegrees', this indicates "
            "whether rotational values are in degrees (yes) or radians (no).\n\n"
        )
        out_file.write("endheader\n")
        out_file.write("time")
        for column in columns:
            out_file.write(f"\t{column}")
        out_file.write("\n")

        for frame_index, pose in enumerate(poses):
            out_file.write(f"{frame_index * timestep_s:.8f}")
            for value in pose:
                out_file.write(f"\t{float(value):.10g}")
            out_file.write("\n")


def list_trials(subject: Any) -> None:
    print(
        json.dumps(
            {
                "massKg": subject.getMassKg(),
                "heightM": subject.getHeightM(),
                "groundForceBodies": list(subject.getGroundForceBodies()),
                "processingPassTypes": [
                    pass_type_name(subject, index)
                    for index in range(subject.getNumProcessingPasses())
                ],
                "trials": [
                    trial_summary(subject, index)
                    for index in range(subject.getNumTrials())
                ],
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b3d", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/addbiomechanics_extracted"))
    parser.add_argument("--trial-index", type=int)
    parser.add_argument("--trial-name")
    parser.add_argument(
        "--processing-pass",
        type=int,
        default=0,
        help="B3D processing pass to export. Pass 0 is typically KINEMATICS.",
    )
    parser.add_argument("--list-trials", action="store_true")
    args = parser.parse_args()

    subject = nimble.biomechanics.SubjectOnDisk(str(args.b3d))
    if args.list_trials:
        list_trials(subject)
        return

    if args.processing_pass >= subject.getNumProcessingPasses():
        raise ValueError(
            f"Processing pass {args.processing_pass} is out of range for "
            f"{subject.getNumProcessingPasses()} subject passes"
        )

    trial_index = select_trial(subject, args.trial_index, args.trial_name)
    if args.processing_pass >= subject.getTrialNumProcessingPasses(trial_index):
        raise ValueError(
            f"Trial {trial_index} has only "
            f"{subject.getTrialNumProcessingPasses(trial_index)} processing passes"
        )

    summary = trial_summary(subject, trial_index)
    trial_slug = slugify(summary["trialName"])
    subject_slug = slugify(args.b3d.stem)
    out_dir = args.out_dir / subject_slug / trial_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    columns = dof_names(subject, args.processing_pass)
    frames = subject.readFrames(
        trial_index,
        0,
        summary["frames"],
        includeSensorData=False,
        includeProcessingPasses=True,
    )
    poses = [
        list(frame.processingPasses[args.processing_pass].pos)
        for frame in frames
    ]

    osim_path = out_dir / f"{subject_slug}_{trial_slug}_kinematics.osim"
    mot_path = out_dir / f"{subject_slug}_{trial_slug}_kinematics.mot"
    metadata_path = out_dir / "b3d_extraction_metadata.json"

    osim_path.write_text(subject.getOpensimFileText(args.processing_pass))
    write_mot(mot_path, columns, poses, summary["timestepS"])
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "gait_nnfe_b3d_kinematics_extraction.v1",
                "sourceB3d": args.b3d.name,
                "sourceB3dPath": str(args.b3d),
                "subject": {
                    "massKg": subject.getMassKg(),
                    "heightM": subject.getHeightM(),
                    "groundForceBodies": list(subject.getGroundForceBodies()),
                },
                "selectedTrial": summary,
                "processingPassIndex": args.processing_pass,
                "processingPassType": pass_type_name(subject, args.processing_pass),
                "outputs": {
                    "osim": osim_path.name,
                    "mot": mot_path.name,
                },
                "dofColumns": columns,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Exported {summary['trialName']} from {args.b3d}")
    print(f"  OSIM: {osim_path}")
    print(f"  MOT:  {mot_path}")
    print(f"  META: {metadata_path}")


if __name__ == "__main__":
    main()
