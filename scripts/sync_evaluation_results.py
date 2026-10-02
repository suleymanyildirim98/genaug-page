#!/usr/bin/env python3
"""Copy a finished GenAug evaluation into the project-page artifact catalog."""

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


MODES = ("content-fidelity", "degradation-transfer")
QUALITATIVE_VISUALIZATION = {
    "content-fidelity": "sample_grid.svg",
    "degradation-transfer": "generated_pairs_grid.svg",
}


def parse_run_name(name):
    match = re.search(r"run(\d+)", name)
    if not match:
        raise ValueError("Could not find a run number in {!r}".format(name))
    return "run{}".format(match.group(1))


def parse_epoch(name):
    match = re.search(r"(?:ckpt|epoch|ep)(last|\d+)$", name)
    if not match:
        raise ValueError(
            "Evaluation directory {!r} must end in ckpt<epoch>, ep<epoch>, "
            "epoch<epoch>, or ckptlast.".format(name)
        )
    return match.group(1)


def copy_artifacts(source, target, mode):
    if not (source / "metrics_summary.json").is_file():
        raise FileNotFoundError("Missing metrics summary: {}".format(source / "metrics_summary.json"))
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source / "metrics_summary.json"), str(target / "metrics_summary.json"))
    for csv_path in source.glob("metrics_*.csv"):
        shutil.copy2(str(csv_path), str(target / csv_path.name))
    destination = target / "visualizations"
    if destination.exists():
        shutil.rmtree(str(destination))
    source_image = source / "visualizations" / QUALITATIVE_VISUALIZATION[mode]
    if source_image.is_file():
        destination.mkdir(parents=True)
        shutil.copy2(str(source_image), str(destination / source_image.name))


def build_manifest(results_root):
    runs = {}
    for run_epoch_dir in sorted(results_root.glob("run*-epoch*")):
        match = re.fullmatch(r"(run\d+)-epoch(.+)", run_epoch_dir.name)
        if not match:
            continue
        run, epoch = match.groups()
        evaluations = {}
        for mode in MODES:
            mode_dir = run_epoch_dir / mode
            if not (mode_dir / "metrics_summary.json").is_file():
                continue
            visualization_dir = mode_dir / "visualizations"
            visualizations = []
            if visualization_dir.is_dir():
                visualizations = sorted(path.name for path in visualization_dir.iterdir() if path.is_file())
            evaluations[mode] = {"visualizations": visualizations}
        if evaluations:
            runs.setdefault(run, {})[epoch] = evaluations
    return {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "runs": runs,
    }


def write_manifest(page_root):
    results_root = page_root / "assets" / "results"
    manifest = build_manifest(results_root)
    output = results_root / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-root", required=True, type=Path)
    parser.add_argument("--source-dir", type=Path, help="Finished content_fidelity or degradation_transfer directory")
    parser.add_argument("--run-name", help="Experiment name, such as genaug_mage_deresnet_run5_mix_fullmask")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--rebuild-manifest", action="store_true")
    args = parser.parse_args()

    page_root = args.page_root.resolve()
    if not (page_root / "index.html").is_file():
        raise FileNotFoundError("Not a project-page root: {}".format(page_root))
    results_root = page_root / "assets" / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    if args.rebuild_manifest:
        if args.source_dir or args.run_name or args.mode:
            parser.error("--rebuild-manifest cannot be combined with source-copy arguments")
    else:
        if not (args.source_dir and args.run_name and args.mode):
            parser.error("--source-dir, --run-name, and --mode are required when copying artifacts")
        source = args.source_dir.resolve()
        if source.name.replace("_", "-") != args.mode:
            raise ValueError("Source directory {} does not match mode {}".format(source, args.mode))
        run = parse_run_name(args.run_name)
        epoch = parse_epoch(source.parent.name)
        target = results_root / "{}-epoch{}".format(run, epoch) / args.mode
        copy_artifacts(source, target, args.mode)
        print("Copied {} results to {}".format(args.mode, target))

    manifest_path, manifest = write_manifest(page_root)
    print("Updated {} with {} runs.".format(manifest_path, len(manifest["runs"])))


if __name__ == "__main__":
    main()
