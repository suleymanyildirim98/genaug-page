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

# Curated, public-facing context for the runs currently shown on the project
# page.  Keep this concise: the page is an experiment navigator, not a dump of
# absolute paths, cluster settings, or tracking identifiers from args.json.
RUN_DESCRIPTIONS = {
    "run3": {
        "title": "BSRGAN+ baseline",
        "summary": "The original GenAug training run, using the BSRGAN+ synthetic-degradation pipeline as the baseline for the first content-fidelity and degradation-transfer snapshots.",
        "purpose": "Establish the behavior of the standard masking recipe with BSRGAN+ degradation training before adding full-mask and mixed-degradation augmentation.",
        "recipe": [
            ["Masking", "Ratio sampled from 0.50 to 1.00 (mean 0.75)"],
            ["Synthetic degradation", "BSRGAN+ pipeline"],
            ["Full-mask training", "Not enabled in the saved configuration"],
        ],
    },
    "run4": {
        "title": "Mixed-degradation + full-mask ablation",
        "summary": "Adds full-mask training and synthetic degradation augmentation while preserving the shared model and optimization recipe.",
        "purpose": "Test whether broader degradation exposure yields clearer condition control without compromising content preservation.",
        "recipe": [
            ["Full-mask probability", "25%"],
            ["Synthetic degradations", "BSRGAN-Light 25% · BSRGAN 35% · BSRGAN+ 40%"],
        ],
    },
    "run5": {
        "title": "Condition-B absolute-target ablation",
        "summary": "Retains the mixed synthetic recipe and full-mask batches while changing the absolute degradation target from source A to the condition-B embedding.",
        "purpose": "Isolate the effect of directly matching the generated degradation to the independent condition image.",
        "recipe": [
            ["Full-mask probability", "25%"],
            ["Synthetic degradations", "BSRGAN-Light 25% · BSRGAN 35% · BSRGAN+ 40%"],
            ["Absolute degradation target", "Condition LR-B embedding"],
        ],
    },
    "run6": {
        "title": "Source-A absolute / condition-B ranking",
        "summary": "Combines mixed synthetic degradations and full-mask batches with source-A absolute matching and condition-B rank supervision.",
        "purpose": "Separate the absolute and ranking degradation objectives to test their respective conditioning roles.",
        "recipe": [
            ["Full-mask probability", "25%"],
            ["Synthetic degradations", "BSRGAN-Light 25% · BSRGAN 35% · BSRGAN+ 40%"],
            ["Absolute degradation target", "Source LR-A embedding"],
            ["Rank target", "Condition LR-B vs. opposite-domain LR-B anchor"],
        ],
    },
}

SHARED_TRAINING_RECIPE = [
    ["Generator", "MAGE ViT-Base / 16"],
    ["Degradation encoder", "MoCo v3 ViT-Base · 256-D representation"],
    ["Training data", "DRealSR ×4"],
    ["Schedule", "1,500 planned epochs · batch size 32"],
]


def parse_experiment_summary(path):
    """Convert the authored Markdown summary into safe, display-ready sections."""
    def public_text(text):
        # The authored notes retain the exact cluster location for local
        # reproducibility; the public page should expose the dataset, not the
        # laboratory filesystem layout.
        return re.sub(r"/scratch/[^\s`]+", "DRealSR ×4 training data", text)

    sections = []
    current = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            current = {"title": line[3:].strip(), "paragraphs": [], "bullets": []}
            sections.append(current)
        elif current and line.startswith("- "):
            current["bullets"].append(public_text(line[2:].strip().replace("`", "")))
        elif current and line:
            current["paragraphs"].append(public_text(line.replace("`", "")))
    return sections


def copy_experiment_summary(experiment_dir, results_root):
    """Copy a run's authored description into the public artifact catalog."""
    experiment_dir = experiment_dir.resolve()
    summary = experiment_dir / "experiment_summary.md"
    if not summary.is_file():
        raise FileNotFoundError("Missing experiment summary: {}".format(summary))
    run = parse_run_name(experiment_dir.name)
    destination = results_root / "experiment_summaries"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(summary), str(destination / "{}.md".format(run)))
    return run


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
    experiment_records = {}
    summary_dir = results_root / "experiment_summaries"
    if summary_dir.is_dir():
        for summary_path in sorted(summary_dir.glob("run*.md")):
            match = re.fullmatch(r"(run\d+)\.md", summary_path.name)
            if match:
                experiment_records[match.group(1)] = parse_experiment_summary(summary_path)
    return {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "run_descriptions": RUN_DESCRIPTIONS,
        "experiment_records": experiment_records,
        "shared_training_recipe": SHARED_TRAINING_RECIPE,
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
    parser.add_argument("--sync-experiment-summary", type=Path, metavar="EXPERIMENT_DIR",
                        help="Copy EXPERIMENT_DIR/experiment_summary.md into the page catalog")
    args = parser.parse_args()

    page_root = args.page_root.resolve()
    if not (page_root / "index.html").is_file():
        raise FileNotFoundError("Not a project-page root: {}".format(page_root))
    results_root = page_root / "assets" / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    if args.sync_experiment_summary:
        if args.source_dir or args.run_name or args.mode or args.rebuild_manifest:
            parser.error("--sync-experiment-summary cannot be combined with other actions")
        run = copy_experiment_summary(args.sync_experiment_summary, results_root)
        print("Copied experiment summary for {}.".format(run))
    elif args.rebuild_manifest:
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
        # A normal evaluation sync also refreshes the authored run description
        # when it is available alongside the output directory.
        experiment_dir = next((parent for parent in source.parents if (parent / "experiment_summary.md").is_file()), None)
        if experiment_dir:
            copy_experiment_summary(experiment_dir, results_root)
        print("Copied {} results to {}".format(args.mode, target))

    manifest_path, manifest = write_manifest(page_root)
    print("Updated {} with {} runs.".format(manifest_path, len(manifest["runs"])))


if __name__ == "__main__":
    main()
