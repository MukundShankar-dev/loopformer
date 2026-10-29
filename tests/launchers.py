"""Stage repository launchers inside a fake repository root for shell tests."""
import shutil
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def stage_launcher(name: str, root: Path) -> str:
    """Copy scripts/launch/<name> under root; launchers cd two levels up to root."""
    target = root / 'scripts/launch' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPOSITORY / 'scripts/launch' / name, target)
    return f'scripts/launch/{name}'
