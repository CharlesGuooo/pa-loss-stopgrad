"""Repository, dataset and upstream-checkout locations.

Everything resolves from the repository root and every value can be overridden
by an environment variable, because none of the large inputs ship with this
repository: nuScenes is downloaded from nuscenes.org under its own licence, the
SparseDrive weights come from the upstream release, and GameFormer is cloned
separately (see ``third_party/setup_gameformer.md``).

    PA_LOSS_REPO_ROOT   repository root                  (default: this checkout)
    PA_LOSS_DATA_ROOT   preprocessed .npz cache          (default: <repo>/data)
    NUSCENES_ROOT       raw nuScenes release             (default: <data>/nuscenes)
    GAMEFORMER_ROOT     MCZhi/GameFormer clone           (default: <repo>/third_party/GameFormer)
    SPARSEDRIVE_ROOT    swc-17/SparseDrive clone         (default: <repo>/third_party/SparseDrive)
"""
import os
from pathlib import Path

# paths.py -> pa_loss_stopgrad -> src -> <repo root>
_HERE = Path(__file__).resolve()
REPO_ROOT = Path(os.environ.get("PA_LOSS_REPO_ROOT", _HERE.parents[2]))

DATA_ROOT = Path(os.environ.get("PA_LOSS_DATA_ROOT", REPO_ROOT / "data"))
NUSCENES_ROOT = Path(os.environ.get("NUSCENES_ROOT", DATA_ROOT / "nuscenes"))

GAMEFORMER_ROOT = Path(
    os.environ.get("GAMEFORMER_ROOT", REPO_ROOT / "third_party" / "GameFormer"))
SPARSEDRIVE_ROOT = Path(
    os.environ.get("SPARSEDRIVE_ROOT", REPO_ROOT / "third_party" / "SparseDrive"))

# Preprocessed GameFormer-format nuScenes shards. The directory name is kept
# from the original pipeline so an existing cache stays usable.
GF_CACHE = DATA_ROOT / "gameformer_nuscenes"


def gf_split(split: str) -> Path:
    """Directory holding the preprocessed shards for ``split``."""
    return GF_CACHE / split


def require(path: Path, what: str, hint: str) -> Path:
    """Fail with an actionable message rather than a bare FileNotFoundError."""
    if not Path(path).exists():
        raise FileNotFoundError(
            "%s not found at %s\n  %s" % (what, path, hint))
    return Path(path)
