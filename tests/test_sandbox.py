"""Adversarial suite for scripts/sandbox: it must pass on a machine before that
machine runs model-written code.

Each attack is paired, where it can be done safely, with a control showing the
same action works on the host, so a pass means the sandbox stopped it rather
than the attack being broken. Resource attacks are paired with a below-limit
run that succeeds, so a pass is not "everything fails".
"""
import errno
import json
import platform
import re
import socket
import struct
import subprocess
import textwrap
import threading
import time
import uuid
from pathlib import Path

import pytest

from scripts.sandbox.runner import HIDDEN_UNDER_USR, Limits, RunResult, run, validate_mounts
from sandbox_marks import needs_sandbox
from scripts.sandbox.seccomp import DENIED_SYSCALLS, build_filter

UNISTD = Path("/usr/include/x86_64-linux-gnu/asm/unistd_64.h")
FAST = Limits(wall_seconds=5, cpu_seconds=5)


def python(code: str, stdin: str = "", limits: Limits = FAST, **kwargs) -> RunResult:
    return run({"main.py": textwrap.dedent(code)}, stdin=stdin, limits=limits, **kwargs)


def lines(result: RunResult) -> list[str]:
    assert result.exit_code == 0, result.stderr
    return result.stdout.splitlines()


# Filter encoding: runs anywhere.

@pytest.mark.skipif(not UNISTD.exists(), reason="needs x86_64 kernel headers")
def test_denied_syscall_numbers_match_kernel_headers():
    defined = dict(re.findall(r"#define __NR_(\w+) (\d+)", UNISTD.read_text()))
    assert {name: int(defined[name]) for name in DENIED_SYSCALLS} == DENIED_SYSCALLS


def evaluate_filter(program: bytes, number: int, first_argument: int = 0, arch: int = 0xC000003E) -> int:
    """Interpret the classic-BPF subset build_filter emits, on one seccomp_data."""
    data = struct.pack("<iIQ6Q", number, arch, 0, first_argument, 0, 0, 0, 0, 0)
    instructions = [struct.unpack("<HBBI", program[i:i + 8]) for i in range(0, len(program), 8)]
    pc, accumulator = 0, 0
    while True:
        code, jump_true, jump_false, k = instructions[pc]
        pc += 1
        if code == 0x20:
            accumulator = struct.unpack_from("<I", data, k)[0]
        elif code in (0x15, 0x35):
            taken = accumulator == k if code == 0x15 else accumulator >= k
            pc += jump_true if taken else jump_false
        elif code == 0x06:
            return k
        else:
            raise AssertionError(f"unexpected opcode {code:#x}")


def test_filter_semantics():
    program = build_filter()
    allow, kill = 0x7FFF0000, 0x80000000
    eperm, eafnosupport = 0x00050000 | errno.EPERM, 0x00050000 | errno.EAFNOSUPPORT
    assert all(evaluate_filter(program, number) == eperm for number in DENIED_SYSCALLS.values())
    assert evaluate_filter(program, 39) == allow  # getpid
    assert evaluate_filter(program, 56, 0x10000000) == allow  # clone is left to --disable-userns
    assert evaluate_filter(program, 39, arch=0x40000003) == kill  # i386 (int 0x80)
    assert evaluate_filter(program, 0x40000000 | 39) == kill  # x32
    for call in (41, 53):  # socket, socketpair
        for family in (1, 2, 10):
            assert evaluate_filter(program, call, family) == allow
        for family in (40, 16, 38, 17, 0, 1 << 32 | 40):  # vsock, netlink, AF_ALG, packet, ...
            assert evaluate_filter(program, call, family) == eafnosupport, family


def test_mount_targets_cannot_shadow_managed_paths(tmp_path):
    for target in ["/usr", "/usr/lib", "/work/x", "/", "docs", "/a/../usr", "/a/"]:
        with pytest.raises(ValueError):
            validate_mounts({target: tmp_path})
    with pytest.raises(FileNotFoundError):
        validate_mounts({"/docs": tmp_path / "missing"})
    validate_mounts({"/docs": tmp_path, "/workspace": tmp_path})


# Normal programs work.

@needs_sandbox
def test_runs_program_with_stdin_exit_code_and_stderr():
    result = python("import sys; data = sys.stdin.read(); print(data.upper()); sys.exit(3)", stdin="abc")
    assert (result.exit_code, result.stdout, result.timed_out, result.oom_killed) == (3, "ABC\n", False, False)
    result = python("raise KeyError('boom')")
    assert result.exit_code == 1 and "KeyError: 'boom'" in result.stderr


@needs_sandbox
def test_standard_library_help_and_source_are_available():
    out = lines(python("""
        import inspect, json, pydoc
        print(len(inspect.getsource(json.dumps)) > 100)
        print('JSONDecoder' in pydoc.render_doc(json, renderer=pydoc.plaintext))
    """))
    assert out == ["True", "True"]


@needs_sandbox
def test_workspace_is_writable_and_extra_mounts_are_read_only(tmp_path):
    (tmp_path / "doc.txt").write_text("reference")
    out = lines(python("""
        import pathlib
        pathlib.Path('/work/a.py').write_text('x = 1')
        pathlib.Path('/tmp/b').write_text('y')
        print(open('/docs/doc.txt').read())
        try:
            open('/docs/new', 'w')
        except OSError as error:
            print(error.errno)
    """, mounts={"/docs": tmp_path}))
    assert out == ["reference", "30"]  # EROFS
    assert sorted(p.name for p in tmp_path.iterdir()) == ["doc.txt"]


# Filesystem: nothing of the host beyond /usr and explicit mounts.

@needs_sandbox
def test_only_expected_paths_are_visible(tmp_path):
    hidden = tmp_path / "hidden_tests.py"
    hidden.write_text("assert solve(1) == 2")
    home = Path.home()
    out = lines(python(f"""
        import json, os
        print(sorted(os.listdir('/')))
        for path in [{str(hidden)!r}, {str(home)!r}, '/mnt/c', '/home', '/root', '/etc/passwd', '/run/WSL',
                     '/init', '/sys', '/tmp/.X11-unix', '/mnt/wslg', '/dev/dxg', '/dev/nvidia0', '/dev/vsock',
                     '/usr/lib/wsl/lib', '/usr/lib/wsl/drivers']:
            print(path, os.path.exists(path))
        print(json.dumps(sorted(os.listdir('/dev'))))
    """))
    assert out[0] == str(['bin', 'code', 'dev', 'lib', 'lib64', 'proc', 'tmp', 'usr', 'work'])
    assert all(line.endswith(" False") for line in out[1:-1]), out
    assert set(json.loads(out[-1])) <= {"console", "core", "fd", "full", "null", "ptmx", "pts", "random", "shm",
                                  "stderr", "stdin", "stdout", "tty", "urandom", "zero"}
    assert hidden.exists() and home.exists()  # controls: both exist on the host


@needs_sandbox
def test_windows_drive_and_executables_unreachable():
    cmd = Path("/mnt/c/Windows/System32/cmd.exe")
    if not cmd.exists():
        pytest.skip("not WSL with a mounted C: drive")
    control = subprocess.run([str(cmd), "/c", "echo", "reachable"], capture_output=True, text=True, timeout=30, cwd="/tmp")
    assert "reachable" in control.stdout  # interop works on the host
    out = lines(python("""
        import os, subprocess
        open('/work/x.exe', 'wb').write(b'MZ' + bytes(510))
        os.chmod('/work/x.exe', 0o755)
        for argv in [['/work/x.exe'], ['/mnt/c/Windows/System32/cmd.exe', '/c', 'echo', 'escaped']]:
            try:
                print(subprocess.run(argv, capture_output=True).returncode)
            except OSError as error:
                print(type(error).__name__)
    """))
    assert all(not line.isdigit() for line in out), out  # neither started


@needs_sandbox
def test_writes_outside_workspace_fail_and_host_is_unchanged(tmp_path):
    (tmp_path / "keep.txt").write_text("original")
    out = lines(python("""
        import errno
        for path in ['/usr/evil', '/code/main.py', '/code/new', '/evil', '/docs/keep.txt', '/proc/sys/kernel/hostname']:
            try:
                open(path, 'w').write('x')
                print(path, 'written')
            except OSError as error:
                print(path, errno.errorcode[error.errno])
    """, mounts={"/docs": tmp_path}))
    assert not any(line.endswith("written") for line in out), out
    assert (tmp_path / "keep.txt").read_text() == "original" and not Path("/usr/evil").exists()


@needs_sandbox
def test_open_file_descriptors_are_only_standard_streams():
    out = lines(python("import os; print(sorted(os.listdir('/proc/self/fd')))"))
    assert out == [str(["0", "1", "2", "3"])]  # 3 is the directory listdir itself opened


@needs_sandbox
def test_environment_holds_no_host_variables(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-secret")
    out = lines(python("import json, os; print(json.dumps(sorted(os.environ)))"))
    assert json.loads(out[0]) == ["HOME", "LANG", "PATH", "PWD", "PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED", "PYTHONUTF8", "TMPDIR"]


# Network.

@needs_sandbox
def test_no_network_not_even_host_loopback():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        socket.create_connection(("127.0.0.1", port), timeout=2).close()  # control: reachable from the host
        out = lines(python(f"""
            import socket
            print(sorted(line.split(':')[0].strip() for line in open('/proc/net/dev').readlines()[2:]))
            for address in [('127.0.0.1', {port}), ('1.1.1.1', 53), ('8.8.8.8', 443)]:
                try:
                    socket.create_connection(address, timeout=2)
                    print('connected')
                except OSError as error:
                    print(type(error).__name__)
            try:
                socket.getaddrinfo('example.com', 443)
                print('resolved')
            except OSError as error:
                print(type(error).__name__)
        """))
    assert out[0] == "['lo']"
    assert "connected" not in out and "resolved" not in out, out


@needs_sandbox
def test_no_vsock_or_other_socket_families():
    """Under WSL2, vsock reaches the Windows host and is not isolated by the network namespace."""
    probe = """
        import socket
        try:
            with socket.socket(40, socket.SOCK_STREAM) as s:  # AF_VSOCK
                s.settimeout(2)
                s.connect((2, 50000))  # the Windows host
                print('connected')
        except OSError as error:
            print(type(error).__name__, error.errno)
    """
    if platform.release().endswith("WSL2"):
        control = subprocess.run(["python3", "-c", textwrap.dedent(probe)], capture_output=True, text=True, timeout=10)
        assert control.stdout.strip() == "connected"  # reachable from the host
    out = lines(python(probe + """
        for family, kind in [(16, socket.SOCK_RAW), (38, socket.SOCK_SEQPACKET), (17, socket.SOCK_RAW), (44, socket.SOCK_RAW)]:
            try:
                socket.socket(family, kind).close()
                print('created', family)
            except OSError as error:
                print(error.errno)
        a, b = socket.socketpair()
        a.sendall(b'unix ok'); print(b.recv(16).decode())
        server = socket.create_server(('127.0.0.1', 0))
        client = socket.create_connection(server.getsockname())
        connection, _ = server.accept(); client.sendall(b'inet ok'); print(connection.recv(16).decode())
    """))
    assert out == [f"OSError {errno.EAFNOSUPPORT}"] + [str(errno.EAFNOSUPPORT)] * 4 + ["unix ok", "inet ok"]


@needs_sandbox
def test_every_mount_is_read_only_except_capped_scratch():
    out = lines(python("""
        import json, os
        mounts = []
        for line in open('/proc/self/mountinfo'):
            fields, rest = line.split(' - ')
            mounts.append([fields.split()[4], fields.split()[5].split(','), rest.split()[0]])
        print(json.dumps(mounts))
        print(json.dumps([name for hidden in HIDDEN if os.path.isdir(hidden) for name in os.listdir(hidden)]))
    """.replace("HIDDEN", repr(HIDDEN_UNDER_USR))))
    writable = {"/proc", "/dev/shm", "/work", "/tmp", "/dev/pts", "/dev/null", "/dev/zero", "/dev/full",
                "/dev/random", "/dev/urandom", "/dev/tty"}
    for target, options, kind in json.loads(out[0]):
        # Host mounts under a hidden directory remain in the table but are covered by an empty tmpfs;
        # uncovering them needs umount, which needs capabilities and is denied by seccomp.
        if any(target.startswith(hidden + "/") for hidden in HIDDEN_UNDER_USR):
            continue
        assert kind not in {"9p", "drvfs", "virtiofs"} and not kind.startswith("fuse"), (target, kind)
        assert "ro" in options or target in writable, (target, options)
        assert target in writable or target in {"/", "/usr", "/code", "/dev", *HIDDEN_UNDER_USR}, target
    assert json.loads(out[1]) == []


@needs_sandbox
def test_inherited_descriptors_lead_nowhere_writable():
    out = lines(python("""
        import errno, os
        print(os.readlink('/proc/self/fd/0').startswith('pipe:'))
        for path in ['/proc/self/fd/1/..', '/dev/evil', '/dev/shm/../evil']:
            try:
                open(path, 'w').write('x')
                print(path, 'written')
            except OSError as error:
                print(path, errno.errorcode[error.errno])
        with open('/proc/self/fd/0', 'w') as handle:  # a pipe: reopening reaches only this run's stdin
            print('reopened')
    """, stdin=""))
    assert out[0] == "True" and not any(line.endswith("written") for line in out[1:4]), out


@needs_sandbox
def test_large_stdin_is_delivered_and_unread_stdin_does_not_hang():
    data = "x" * (4 << 20)
    assert lines(python("import sys; print(len(sys.stdin.read()))", stdin=data)) == [str(len(data))]
    result = python("print('ignored stdin')", stdin=data)
    assert result.stdout == "ignored stdin\n" and not result.timed_out


@needs_sandbox
def test_seccomp_cannot_be_bypassed_through_other_syscall_abis():
    int80_getpid = """
        import ctypes, mmap
        page = mmap.mmap(-1, 4096, prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
        page.write(bytes([0xB8, 20, 0, 0, 0, 0xCD, 0x80, 0xC3]))  # mov eax, 20 (i386 getpid); int 0x80; ret
        address = ctypes.addressof(ctypes.c_char.from_buffer(page))
        print(ctypes.CFUNCTYPE(ctypes.c_int)(address)())
    """
    control = subprocess.run(["python3", "-c", textwrap.dedent(int80_getpid)], capture_output=True, text=True)
    if control.returncode == 0:
        assert int(control.stdout) > 0  # the i386 ABI works on the host
    for code in [int80_getpid, "import ctypes; print(ctypes.CDLL(None).syscall(0x40000000 | 39))"]:
        result = python(code)
        assert result.exit_code == 128 + 31 and result.stdout == "", result  # SIGSYS: killed, nothing ran


# Processes: the program cannot see, signal or outlive into the host.

@needs_sandbox
def test_host_processes_invisible_and_unkillable():
    marker = subprocess.Popen(["sleep", "30"])
    try:
        out = lines(python(f"""
            import os, signal
            print(max(int(p) for p in os.listdir('/proc') if p.isdigit()))
            try:
                os.kill({marker.pid}, signal.SIGKILL)
                print('killed')
            except OSError as error:
                print(type(error).__name__)
        """))
        assert marker.poll() is None  # still running
        assert int(out[0]) < 10 and out[1] == "ProcessLookupError", out
    finally:
        marker.kill()


@needs_sandbox
def test_background_processes_die_with_the_run():
    marker = f"lfsbx-marker-{uuid.uuid4().hex}"
    result = python(f"""
        import subprocess
        subprocess.Popen(['python3', '-c', 'import time; time.sleep(300)', '{marker}'], start_new_session=True)
        print('detached')
    """)
    assert result.stdout == "detached\n"
    time.sleep(0.5)
    assert subprocess.run(["pgrep", "-f", marker], capture_output=True).stdout == b""
    state = subprocess.run(["systemctl", "--user", "is-active", result.unit], capture_output=True, text=True).stdout
    assert state.strip() in {"inactive", "unknown", ""}


@needs_sandbox
def test_denied_syscalls_fail_with_eperm():
    control = subprocess.run(["python3", "-c", "import ctypes; print(ctypes.CDLL(None, use_errno=True).syscall(135, 0xffffffff))"],
                             capture_output=True, text=True)
    assert control.stdout.strip() != "-1"  # personality() query succeeds on the host
    out = lines(python(f"""
        import ctypes, errno
        libc = ctypes.CDLL(None, use_errno=True)
        for name, number in sorted({DENIED_SYSCALLS!r}.items()):
            ctypes.set_errno(0)
            code = libc.syscall(number, 0, 0, 0, 0, 0)
            print(name, code, errno.errorcode.get(ctypes.get_errno()))
        print('getpid', libc.syscall(39) > 0)
    """))
    assert out[-1] == "getpid True"
    assert all(line.endswith(" -1 EPERM") for line in out[:-1]), [line for line in out if "EPERM" not in line]


@needs_sandbox
def test_cannot_create_namespaces_or_gain_privileges():
    out = lines(python("""
        import json, os, subprocess
        print(os.getuid(), os.geteuid())
        print(json.dumps([line for line in open('/proc/self/status') if line.startswith(('CapEff', 'NoNewPrivs', 'Seccomp:'))]))
        for argv in [['unshare', '-U', 'true'], ['unshare', '-r', 'true']]:
            try:
                print(subprocess.run(argv, capture_output=True).returncode)
            except OSError as error:
                print(type(error).__name__)
        # clone() is not in the seccomp denylist, so only --disable-userns stops this one.
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        pid = libc.syscall(56, 0x10000000 | 17, 0, 0, 0, 0)  # clone(CLONE_NEWUSER | SIGCHLD), fork-like
        if pid == 0:
            os._exit(0)
        print('clone', 'created' if pid > 0 else 'refused')
    """))
    assert out[0] == "65534 65534"
    status = json.loads(out[1])
    assert "CapEff:\t0000000000000000\n" in status and "NoNewPrivs:\t1\n" in status and "Seccomp:\t2\n" in status
    assert all(code != "0" for code in out[2:4]) and out[4] == "clone refused", out


# Resource limits: each paired with a below-limit run that succeeds.

@needs_sandbox
def test_infinite_loop_and_sleep_are_stopped_at_the_wall_limit():
    limits = Limits(wall_seconds=2, cpu_seconds=10)
    for code in ["while True: pass", "import time; time.sleep(60)"]:
        result = python(code, limits=limits)
        assert result.timed_out and result.exit_code != 0 and result.wall_seconds < 3, result
    assert not python("import time; time.sleep(0.5)", limits=limits).timed_out


@needs_sandbox
def test_cpu_limit_kills_a_spinning_process():
    result = python("while True: pass", limits=Limits(wall_seconds=10, cpu_seconds=1))
    assert not result.timed_out and result.exit_code == 128 + 24 and result.wall_seconds < 4, result  # SIGXCPU


@needs_sandbox
def test_memory_hog_is_oom_killed_without_swap():
    limits = Limits(memory_bytes=256 << 20)
    assert lines(python("x = bytearray(64 << 20); print(len(x))", limits=limits)) == [str(64 << 20)]
    result = python("x = bytearray(1 << 30)", limits=limits)
    assert result.oom_killed and result.exit_code != 0, result


@needs_sandbox
def test_tmpfs_fill_is_capped():
    limits = Limits(tmpfs_bytes=8 << 20, file_bytes=64 << 20)
    out = lines(python("""
        import errno
        for path in ['/work/fill', '/tmp/fill', '/dev/shm/fill']:
            written = 0
            try:
                with open(path, 'wb') as handle:
                    while True:
                        handle.write(bytes(1 << 20)); handle.flush(); written += 1
            except OSError as error:
                print(errno.errorcode[error.errno], written)
    """, limits=limits))
    for line in out:
        code, megabytes = line.split()
        assert code == "ENOSPC" and 4 <= int(megabytes) <= 8, out
    assert len(out) == 3


@needs_sandbox
def test_process_count_is_capped():
    out = lines(python("""
        import os, time
        children = []
        try:
            for _ in range(500):
                pid = os.fork()
                if pid == 0:
                    time.sleep(30)
                    os._exit(0)
                children.append(pid)
        except BlockingIOError:
            pass
        print(len(children))
    """, limits=Limits(processes=32)))
    assert 20 <= int(out[0]) < 32


@needs_sandbox
def test_fork_bomb_is_contained():
    start = time.perf_counter()
    result = python("""
        import os
        while True:
            try:
                os.fork()
            except OSError:
                pass
    """, limits=Limits(wall_seconds=3, processes=32))
    assert result.timed_out and time.perf_counter() - start < 10
    assert subprocess.run(["systemctl", "--user", "is-active", result.unit], capture_output=True, text=True).stdout.strip() != "active"


@needs_sandbox
def test_output_flood_is_truncated_and_bounded_on_disk():
    limits = Limits(file_bytes=1 << 20, output_bytes=1000)
    result = python("import sys\nwhile True: sys.stdout.write('x' * 4096)", limits=limits)
    # CPython ignores SIGXFSZ, so the write fails with EFBIG instead of the process dying.
    assert result.exit_code == 1 and not result.timed_out and "[Errno 27] File too large" in result.stderr
    assert result.stdout_truncated and len(result.stdout) == 1000
    small = python("print('y' * 10)", limits=limits)
    assert small.stdout == "y" * 10 + "\n" and not small.stdout_truncated


@needs_sandbox
def test_runs_are_independent():
    python("open('/work/state', 'w').write('left behind'); open('/tmp/state', 'w').write('x')")
    out = lines(python("import os; print(os.listdir('/work'), os.listdir('/tmp'))"))
    assert out == ["[] []"]


@needs_sandbox
def test_concurrent_runs_do_not_interfere():
    results = {}

    def worker(index: int) -> None:
        results[index] = python(f"print({index})")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert {i: r.stdout for i, r in results.items()} == {i: f"{i}\n" for i in range(8)}
