"""Run untrusted Python inside bubblewrap, inside a systemd cgroup scope.

Layers, outermost first:
- `systemd-run --user --scope`: a transient cgroup with memory (no swap),
  task-count and CPU-quota limits. Every process the program starts stays in
  it, so a timeout kills the whole scope rather than one process.
- `prlimit`: per-process CPU time, file size, open files, no core dumps.
- `bwrap`: new user, PID, network, IPC, UTS and cgroup namespaces; no nested
  user namespaces; no capabilities; no-new-privileges; a seccomp denylist
  (scripts/sandbox/seccomp.py); a cleared environment.

The program sees only `/usr` (read-only, so the interpreter, the standard
library and its source), the files it was given at `/code` (read-only), any
extra read-only `mounts`, and fresh size-capped tmpfs at `/work`, `/tmp` and
`/dev/shm`. There is no `/home`, `/mnt`, `/etc`, `/sys`, `/run` or host `/tmp`,
so no Windows drives, WSL interop sockets, GPU devices or hidden tests, and
`/usr/lib/wsl` (Windows driver files over 9p) and `/usr/lib/modules` are
covered by empty tmpfs.
Everything else, including bwrap's own root and `/dev`, is read-only. Stdin
arrives through a pipe. Nothing the program does writes to the host except its
stdout and stderr, which go to size-capped files that are read back here and
truncated.

tests/test_sandbox.py is the adversarial suite; it must pass on a machine
before that machine runs model-written code.
"""
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from scripts.sandbox.seccomp import build_filter

RESERVED = {"/", "/usr", "/bin", "/lib", "/lib64", "/proc", "/dev", "/code", "/work", "/tmp"}
# Host directories under /usr that must not be visible: WSL's Windows driver store (9p), GPU libraries and
# kernel modules. Each is covered by an empty read-only tmpfs where it exists.
HIDDEN_UNDER_USR = ("/usr/lib/wsl", "/usr/lib/modules")
SANDBOX_ENVIRONMENT = {
    "PATH": "/usr/bin", "HOME": "/work", "TMPDIR": "/tmp", "LANG": "C.UTF-8",
    "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
}
SANDBOX_UID = 65534
POLL_SECONDS = 0.005


@dataclass(frozen=True)
class Limits:
    wall_seconds: float = 10.0
    cpu_seconds: int = 10  # per process; SIGXCPU at the limit, SIGKILL one second later
    memory_bytes: int = 512 << 20  # whole scope, including tmpfs pages; swap is disabled
    processes: int = 64  # tasks (processes and threads) in the scope
    cpus: int = 1  # CPU quota of the whole scope, in cores
    file_bytes: int = 16 << 20  # largest file any process may write, stdout and stderr included
    tmpfs_bytes: int = 64 << 20  # each of /work, /tmp and /dev/shm
    open_files: int = 256
    output_bytes: int = 64 << 10  # stdout and stderr returned to the caller, each

    def __post_init__(self) -> None:
        if self.wall_seconds <= 0 or self.cpu_seconds < 1 or self.processes < 2 or self.cpus < 1:
            raise ValueError(f"Limits too small to run anything: {self}")


@dataclass(frozen=True)
class RunResult:
    """What the host observed. `exit_code` follows bwrap: 128 + N when the
    program was killed by signal N. `stdout` keeps the head of the output and
    `stderr` the tail, where a traceback ends. Both are untrusted text."""

    exit_code: int
    timed_out: bool
    oom_killed: bool
    wall_seconds: float
    cpu_seconds: float
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    unit: str = field(repr=False)


def check_host() -> None:
    """Fail with the missing piece rather than with an obscure bwrap error."""
    for tool in ("bwrap", "systemd-run", "systemctl", "prlimit"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not found; see docs/setup.md")
    for variable in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"):
        if variable not in os.environ:
            raise RuntimeError(f"{variable} unset: the systemd user manager is needed for cgroup limits")


def bwrap_arguments(mounts: Mapping[str, Path], limits: Limits, seccomp_fd: int, code_dir: Path) -> list[str]:
    arguments = [
        "bwrap", "--unshare-all", "--unshare-user", "--disable-userns",
        "--uid", str(SANDBOX_UID), "--gid", str(SANDBOX_UID), "--hostname", "sandbox",
        "--die-with-parent", "--new-session", "--cap-drop", "ALL", "--clearenv",
        "--seccomp", str(seccomp_fd),
        "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
        "--proc", "/proc", "--dev", "/dev",
        "--size", str(limits.tmpfs_bytes), "--tmpfs", "/dev/shm",
        "--size", str(limits.tmpfs_bytes), "--tmpfs", "/work",
        "--size", str(limits.tmpfs_bytes), "--tmpfs", "/tmp",
        "--ro-bind", str(code_dir), "/code",
    ]
    hidden = [path for path in HIDDEN_UNDER_USR if Path(path).is_dir()]
    for path in hidden:
        arguments += ["--tmpfs", path]
    for target, source in sorted(mounts.items()):
        arguments += ["--ro-bind", str(source), target]
    for name, value in SANDBOX_ENVIRONMENT.items():
        arguments += ["--setenv", name, value]
    # bwrap's root and /dev are uncapped tmpfs; only the capped ones should take writes.
    for path in [*hidden, "/dev", "/"]:
        arguments += ["--remount-ro", path]
    return [*arguments, "--chdir", "/work"]


def validate_mounts(mounts: Mapping[str, Path]) -> None:
    for target, source in mounts.items():
        path = PurePosixPath(target)
        if not path.is_absolute() or ".." in path.parts or str(path) != target:
            raise ValueError(f"Mount target must be a normalised absolute path: {target!r}")
        if target in RESERVED or any(target.startswith(reserved + "/") for reserved in RESERVED - {"/"}):
            raise ValueError(f"Mount target {target!r} overlaps a sandbox-managed path")
        if not Path(source).exists():
            raise FileNotFoundError(f"Mount source {source} does not exist")


def write_files(directory: Path, files: Mapping[str, str]) -> None:
    for name, text in files.items():
        if not name or "/" in name or name in {".", ".."}:
            raise ValueError(f"File names must be plain names: {name!r}")
        (directory / name).write_text(text)


def read_capped(path: Path, limit: int, tail: bool = False) -> tuple[str, bool]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(max(0, size - limit) if tail else 0)
        data = handle.read(limit)
    return data.decode("utf-8", errors="replace"), size > limit


def scope_result(unit: str) -> str:
    """systemd's Result for the finished scope (`oom-kill` after an OOM kill).
    A failed scope stays loaded until reset, so reset it here."""
    shown = subprocess.run(["systemctl", "--user", "show", unit, "-p", "Result", "--value"],
                           capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(["systemctl", "--user", "reset-failed", unit], capture_output=True)
    return shown


def run(files: Mapping[str, str], entry: str = "main.py", stdin: str = "", limits: Limits = Limits(),
        mounts: Mapping[str, Path] = {}, python: str = "/usr/bin/python3") -> RunResult:
    """Run `python -I /code/<entry>` with `files` in /code and `stdin` on standard input."""
    check_host()
    if entry not in files:
        raise ValueError(f"entry {entry!r} is not among the files")
    validate_mounts(mounts)
    unit = f"lfsbx-{uuid.uuid4().hex}.scope"
    with tempfile.TemporaryDirectory(prefix="lfsbx-") as scratch:
        scratch = Path(scratch)
        code_dir = scratch / "code"
        code_dir.mkdir()
        write_files(code_dir, files)
        (scratch / "filter.bpf").write_bytes(build_filter())
        # Stdin is a pipe, not a file: the program could reopen a file through /proc/self/fd/0 for writing.
        with (open(scratch / "stdout", "wb") as stdout_file,
              open(scratch / "stderr", "wb") as stderr_file, open(scratch / "filter.bpf", "rb") as seccomp_file):
            argv = [
                "systemd-run", "--user", "--scope", "--quiet", f"--unit={unit}",
                "-p", f"MemoryMax={limits.memory_bytes}", "-p", "MemorySwapMax=0",
                "-p", f"TasksMax={limits.processes}", "-p", f"CPUQuota={100 * limits.cpus}%",
                # Backstop if this process dies before it can enforce the wall limit.
                "-p", f"RuntimeMaxSec={int(limits.wall_seconds) + 5}",
                "prlimit", f"--cpu={limits.cpu_seconds}:{limits.cpu_seconds + 1}",
                f"--fsize={limits.file_bytes}", f"--nofile={limits.open_files}", "--core=0", "--",
                *bwrap_arguments(mounts, limits, seccomp_file.fileno(), code_dir),
                "--", python, "-I", f"/code/{entry}",
            ]
            environment = {"PATH": "/usr/bin:/bin", **{k: os.environ[k] for k in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")}}
            start = time.perf_counter()
            process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=stdout_file, stderr=stderr_file, env=environment,
                                       pass_fds=(seccomp_file.fileno(),), start_new_session=True)
            feeder = threading.Thread(target=feed, args=(process.stdin, stdin.encode()), daemon=True)
            feeder.start()
            status, usage, timed_out = wait(process, start + limits.wall_seconds, unit)
            feeder.join()
            wall = time.perf_counter() - start
        stdout, stdout_truncated = read_capped(scratch / "stdout", limits.output_bytes)
        stderr, stderr_truncated = read_capped(scratch / "stderr", limits.output_bytes, tail=True)
    return RunResult(
        exit_code=os.waitstatus_to_exitcode(status), timed_out=timed_out, oom_killed=scope_result(unit) == "oom-kill",
        wall_seconds=wall, cpu_seconds=usage.ru_utime + usage.ru_stime, stdout=stdout, stderr=stderr,
        stdout_truncated=stdout_truncated, stderr_truncated=stderr_truncated, unit=unit)


def feed(pipe, data: bytes) -> None:
    """Write stdin in a thread so a program that never reads cannot block the wait loop."""
    try:
        with pipe:  # closes the descriptor even if the final flush fails
            pipe.write(data)
    except BrokenPipeError:  # the program exited or closed stdin without reading it all
        pass


def wait(process: subprocess.Popen, deadline: float, unit: str):
    """Wait for the scope's root process, killing the whole scope at the deadline.
    Uses wait4 for CPU time of the program and every descendant it reaped."""
    while True:
        pid, status, usage = os.wait4(process.pid, os.WNOHANG)
        if pid:
            process.returncode = os.waitstatus_to_exitcode(status)
            return status, usage, False
        if time.perf_counter() >= deadline:
            break
        time.sleep(POLL_SECONDS)
    subprocess.run(["systemctl", "--user", "kill", "--signal=SIGKILL", unit], capture_output=True)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:  # exited between the last poll and the kill
        pass
    _, status, usage = os.wait4(process.pid, 0)
    process.returncode = os.waitstatus_to_exitcode(status)
    return status, usage, True
