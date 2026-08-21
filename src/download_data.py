"""Download and verify the exact public Kaggle dataset used by this project."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_HANDLE = "atharvaingle/crop-recommendation-dataset"
DATASET_FILENAME = "Crop_recommendation.csv"
DEFAULT_DESTINATION = PROJECT_ROOT / "data" / DATASET_FILENAME
DEFAULT_CACHE = PROJECT_ROOT / "work" / "kagglehub"


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest for *path*."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_dataset(
    destination: str | Path = DEFAULT_DESTINATION,
    cache_dir: str | Path = DEFAULT_CACHE,
    *,
    force_download: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Download the latest public dataset version and copy it into ``data/``.

    An existing identical destination is kept untouched. A different existing
    file is never overwritten unless the caller explicitly sets ``overwrite``.
    """
    destination_path = Path(destination).resolve()
    cache_path = Path(cache_dir).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    # KaggleHub reads this documented environment variable for its cache root.
    os.environ["KAGGLEHUB_CACHE"] = str(cache_path)
    import kagglehub  # Imported after setting the project-local cache location.

    downloaded_dir = Path(
        kagglehub.dataset_download(DATASET_HANDLE, force_download=force_download)
    )
    source_path = downloaded_dir / DATASET_FILENAME
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Kaggle dataset did not contain {DATASET_FILENAME}: {downloaded_dir}"
        )

    source_hash = sha256_file(source_path)
    if destination_path.exists():
        destination_hash = sha256_file(destination_path)
        if destination_hash == source_hash:
            return {
                "dataset_handle": DATASET_HANDLE,
                "source_path": str(source_path),
                "destination_path": str(destination_path),
                "sha256": source_hash,
                "identical": True,
                "copied": False,
            }
        if not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite a different dataset at {destination_path}. "
                "Inspect it first or rerun with --overwrite."
            )

    shutil.copy2(source_path, destination_path)
    destination_hash = sha256_file(destination_path)
    if destination_hash != source_hash:
        raise OSError("Dataset checksum changed while copying from KaggleHub.")

    return {
        "dataset_handle": DATASET_HANDLE,
        "source_path": str(source_path),
        "destination_path": str(destination_path),
        "sha256": destination_hash,
        "identical": True,
        "copied": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ask KaggleHub to refresh its cached public dataset version.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing non-identical destination after inspection.",
    )
    args = parser.parse_args()
    result = download_dataset(
        force_download=args.force_download,
        overwrite=args.overwrite,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

