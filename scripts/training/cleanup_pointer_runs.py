"""Preview or remove the explicitly retired pointer runs; preserve the 30k baseline."""

import argparse
from pathlib import Path
import shutil

from rich.console import Console
from rich.table import Table


LEGACY_RUNS = {
    "20260910T214347.326698Z", "20260910T220130.926886Z",
    "20260911T003442.178029Z", "depth6-seed17",
}
BASELINE = "depth6-fresh30k-seed37-batch4"
KEEP_STEPS = {"step-002250", "step-002500", "step-002750", "step-003000", "step-003250", "step-003750"}
LOOPBALANCED = "depth6-loopbalanced-seed37"
LOOPBALANCED_EVALS = (
    "20260929T232347.929385Z-depth6-loopbalanced-seed37-step-002500",
    "20260929T232646.364715Z-depth6-loopbalanced-seed37-step-002500",
    "20260929T233210.473739Z-depth6-loopbalanced-seed37-step-003250",
    "20260929T233501.570440Z-depth6-loopbalanced-seed37-step-003250",
    "20260929T234030.258696Z-depth6-loopbalanced-seed37-step-003750",
    "20260929T234322.963417Z-depth6-loopbalanced-seed37-step-003750",
)


def cleanup_candidates(root: Path, *, loopbalanced_only: bool = False) -> list[Path]:
    """Only named historical runs and redundant baseline checkpoint directories.

    Keep the baseline logs/evaluations, ordinary-Qwen baseline, all datasets,
    Stage 0 evidence, and every new or unknown run. Never follow symlink targets.
    """
    if loopbalanced_only:
        paths = [root / "models/stage1_pointer" / LOOPBALANCED]
        paths.extend(root / "eval/pointer_loops" / name for name in LOOPBALANCED_EVALS)
        return _validate_candidates(root, paths)
    paths = [root / "models/stage1_pointer" / name for name in sorted(LEGACY_RUNS)]
    for family in ("pointer_loops", "pointer_task"):
        parent = root / "eval" / family
        if parent.exists():
            paths.extend(p for p in parent.iterdir() if any(name in p.name for name in LEGACY_RUNS))
    paths.append(root / "eval/pointer_depth_comparison/20260911T020824.842584Z")
    baseline = root / "models/stage1_pointer" / BASELINE
    if baseline.exists():
        paths.extend(p for p in baseline.glob("step-*") if p.name not in KEEP_STEPS
                     and len(p.name) == 11 and p.name[5:].isdigit())
    return _validate_candidates(root, paths)


def _validate_candidates(root: Path, paths: list[Path]) -> list[Path]:
    result = sorted(p for p in paths if p.exists() or p.is_symlink())
    for p in result:
        if p.is_symlink() or not p.is_dir() or p.resolve() != root.resolve() / p.relative_to(root):
            raise ValueError(f"Refusing unexpected or symlinked cleanup target: {p}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Delete listed directories; default is preview only")
    parser.add_argument("--loopbalanced-only", action="store_true", help="Only the retired loop-balanced run and its six named evaluations, including ignored weights")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    paths = cleanup_candidates(root, loopbalanced_only=args.loopbalanced_only)
    console = Console()
    table = Table(title="Retired pointer artifacts" + (" · DELETE" if args.apply else " · preview"))
    table.add_column("Directory")
    table.add_column("MiB", justify="right")
    total = 0
    for path in paths:
        size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())
        total += size
        table.add_row(str(path.relative_to(root)), f"{size / 2**20:.2f}")
    console.print(table)
    console.print(f"{len(paths)} directories · {total / 2**20:.2f} MiB")
    if args.apply:
        for path in paths:
            shutil.rmtree(path)
        console.print("[green]Removed listed artifacts. Other runs and retained baseline evidence are unchanged.[/green]")
    else:
        console.print("[yellow]Preview only. Add --apply to delete. Do not run while an affected run is active.[/yellow]")


if __name__ == "__main__":
    main()
