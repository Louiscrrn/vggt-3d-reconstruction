import json
import shutil
import argparse
from pathlib import Path
import imageio.v2 as imageio
import numpy as np


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def extract_sequences(split_file: Path, split_name: str):
    with open(split_file, "r") as f:
        data = json.load(f)

    if split_name not in data:
        raise ValueError(f"Split '{split_name}' not found in {split_file}")

    entries = data[split_name]
    sequences = set(entry[0] for entry in entries)
    return sequences


def count_frames(sequence_path: Path):
    images_folder = sequence_path / "images"
    if not images_folder.exists():
        return 0
    return len(list(images_folder.glob("*.jpg")))


def has_valid_depth(sequence_path: Path):
    """
    Vérifie qu'au moins une depth image contient des valeurs non nulles.
    """
    depths_folder = sequence_path / "depths"
    if not depths_folder.exists():
        return False

    for depth_file in depths_folder.glob("*.png"):
        depth = imageio.imread(depth_file)
        if np.max(depth) > 0:
            return True

    return False


def copy_sequence(src_seq: Path, dst_seq: Path):
    dst_seq.mkdir(parents=True, exist_ok=True)

    for folder in ["images", "masks", "depths", "depth_masks"]:
        src_folder = src_seq / folder
        if src_folder.exists():
            shutil.copytree(src_folder, dst_seq / folder)

    # on copie uniquement les json locaux s’il y en a
    for file in src_seq.glob("*.json"):
        shutil.copy(file, dst_seq / file.name)


# ------------------------------------------------------------
# Processing category
# ------------------------------------------------------------

def process_category(src_category: Path, dst_root: Path, split_name: str):

    print(f"\nProcessing {src_category.name}")

    split_file = src_category / "set_lists" / "set_lists_manyview_test_0.json"
    if not split_file.exists():
        print("  No split file found. Skipping.")
        return 0, 0

    sequences = extract_sequences(split_file, split_name)

    dst_category = dst_root / src_category.name
    dst_category.mkdir(parents=True, exist_ok=True)

    # copier annotations globales
    for ann in ["frame_annotations.jgz", "sequence_annotations.jgz"]:
        src_ann = src_category / ann
        if src_ann.exists():
            shutil.copy(src_ann, dst_category / ann)

    kept_sequences = 0
    kept_frames = 0

    for item in src_category.iterdir():
        if not item.is_dir():
            continue

        if item.name in ["set_lists", "eval_batches"]:
            continue

        # doit être dans le split
        if item.name not in sequences:
            continue

        # doit avoir depth valide
        if not has_valid_depth(item):
            print(f"  Skipping {item.name} (no valid depth)")
            continue

        print(f"  Keeping {item.name}")

        kept_sequences += 1
        kept_frames += count_frames(item)

        copy_sequence(item, dst_category / item.name)

    print(f"  → Kept {kept_sequences} sequences | {kept_frames} frames")

    return kept_sequences, kept_frames


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to keep",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="../data/co3D_raw",
        help="Path to raw CO3D dataset",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="../data/co3D_clean",
        help="Path to output dataset",
    )

    args = parser.parse_args()

    SOURCE_ROOT = Path(args.source)
    TARGET_ROOT = Path(args.target) / args.split

    TARGET_ROOT.mkdir(parents=True, exist_ok=True)

    total_sequences = 0
    total_frames = 0

    for category in SOURCE_ROOT.iterdir():
        if category.is_dir():
            seq_count, frame_count = process_category(
                category, TARGET_ROOT, args.split
            )
            total_sequences += seq_count
            total_frames += frame_count

    print("\n===== FINAL SUMMARY =====")
    print(f"Total sequences kept: {total_sequences}")
    print(f"Total frames kept: {total_frames}")