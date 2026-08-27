import json
from dataclasses import dataclass
from pathlib import Path


def _is_source_data_root(path):
    return (path / "01_core").is_dir() and (path / "02_uniprot").is_dir()


def _resolve_source_data(root, value):
    candidate = (root / value).resolve()
    if candidate.exists() or Path(value).is_absolute():
        return candidate

    # A git worktree may be nested below the repository while datasets remain
    # beside the primary checkout. Locate that stable project-level directory.
    for ancestor in (root, *root.parents):
        for possible in (ancestor / "datasets", ancestor.parent / "datasets"):
            possible = possible.resolve()
            if _is_source_data_root(possible):
                return possible
    return candidate


@dataclass(frozen=True)
class MultimodalConfig:
    source_data: Path
    processed_data: Path
    artifacts: Path
    seed: int
    window: int
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    modality_dropout: float = 0.2
    embedding_dim: int = 64
    pu_prior_multiplier: float = 2.0

    @classmethod
    def load(cls, path, repository_root=None):
        path = Path(path)
        values = json.loads(path.read_text(encoding="utf-8"))
        root = Path(repository_root or path.resolve().parents[2]).resolve()
        values["source_data"] = _resolve_source_data(root, values["source_data"])
        for name in ("processed_data", "artifacts"):
            values[name] = (root / values[name]).resolve()
        config = cls(**values)
        total = config.train_fraction + config.validation_fraction + config.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError("train, validation, and test fractions must sum to 1")
        if config.window % 2 != 1:
            raise ValueError("sequence window must be odd")
        return config

    def ensure_output_directories(self):
        self.processed_data.mkdir(parents=True, exist_ok=True)
        self.artifacts.mkdir(parents=True, exist_ok=True)
