"""Run the four single-modality nnPU ablations sequentially on one GPU."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ABLATIONS = (
    ("去序列", "sequence"),
    ("去结构", "structure"),
    ("去PPI网络", "network"),
    ("去质谱蛋白质组", "proteomics"),
)
SEEDS = (11, 23, 37, 51, 73)


def build_commands(python: Path, config: Path, output_root: Path):
    """Build reproducible command lines for the four modality ablations."""
    common = [
        str(python), "-m", "src.multimodal.train", "--config", str(config),
        "--seeds", *(str(seed) for seed in SEEDS), "--epochs", "200", "--patience", "30",
        "--batch-size", "512", "--objective", "nnpu", "--device", "cuda",
        "--transformer-layers", "2", "--modality-dropout", "0", "--network-variant", "complete",
    ]
    return [
        (name, [*common, "--output-root", str(output_root / name), "--disable-modalities", modality])
        for name, modality in ABLATIONS
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--config", type=Path, default=Path("configs/multimodal/default.json"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)

    args.output_root.mkdir(parents=True, exist_ok=True)
    log_path = args.output_root / "训练日志.txt"
    with log_path.open("a", encoding="utf-8") as log:
        for name, command in build_commands(args.python, args.config, args.output_root):
            log.write(f"开始 {name}\n")
            log.flush()
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise SystemExit(f"{name} 训练失败，退出码 {result.returncode}；详见 {log_path}")
            log.write(f"完成 {name}\n")
            log.flush()


if __name__ == "__main__":
    main()
