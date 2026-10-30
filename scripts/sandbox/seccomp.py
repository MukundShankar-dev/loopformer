"""A classic-BPF seccomp filter for bubblewrap's `--seccomp FD`.

Default-allow with a denylist: bubblewrap's namespaces already hide the host,
so the filter only removes kernel attack surface that a Python solution never
needs. Denied calls fail with EPERM, so a program that probes them sees an
ordinary error. Any other architecture or the x32 ABI is killed, because the
numbers below are x86_64 numbers and would not match.

Sockets are limited to Unix, IPv4 and IPv6 families. The network namespace
already isolates those, but not AF_VSOCK: under WSL2 a vsock connection reaches
services on the Windows host. Other families (netlink, AF_ALG, ...) are kernel
attack surface the program does not need; they fail with EAFNOSUPPORT.
"""
import errno
import struct

# x86_64 numbers, checked against asm/unistd_64.h in tests/test_sandbox.py.
DENIED_SYSCALLS = {
    "ptrace": 101, "syslog": 103, "uselib": 134, "personality": 135, "vhangup": 153,
    "modify_ldt": 154, "pivot_root": 155, "adjtimex": 159, "chroot": 161, "acct": 163,
    "settimeofday": 164, "mount": 165, "umount2": 166, "swapon": 167, "swapoff": 168,
    "reboot": 169, "iopl": 172, "ioperm": 173, "init_module": 175, "delete_module": 176,
    "quotactl": 179, "lookup_dcookie": 212, "clock_settime": 227, "kexec_load": 246,
    "add_key": 248, "request_key": 249, "keyctl": 250, "unshare": 272, "perf_event_open": 298,
    "fanotify_init": 300, "name_to_handle_at": 303, "open_by_handle_at": 304,
    "clock_adjtime": 305, "setns": 308, "process_vm_readv": 310, "process_vm_writev": 311,
    "kcmp": 312, "finit_module": 313, "kexec_file_load": 320, "bpf": 321, "userfaultfd": 323,
    "io_uring_setup": 425, "io_uring_enter": 426, "io_uring_register": 427, "open_tree": 428,
    "move_mount": 429, "fsopen": 430, "fsconfig": 431, "fsmount": 432, "fspick": 433,
    "pidfd_getfd": 438, "mount_setattr": 442, "quotactl_fd": 443,
}

SOCKET_CALLS = (41, 53)  # socket, socketpair
ALLOWED_SOCKET_FAMILIES = (1, 2, 10)  # AF_UNIX, AF_INET, AF_INET6

AUDIT_ARCH_X86_64 = 0xC000003E
X32_SYSCALL_BIT = 0x40000000
LOAD_WORD = 0x20  # BPF_LD | BPF_W | BPF_ABS
JUMP_EQUAL = 0x15  # BPF_JMP | BPF_JEQ | BPF_K
JUMP_AT_LEAST = 0x35  # BPF_JMP | BPF_JGE | BPF_K
RETURN = 0x06  # BPF_RET | BPF_K
RET_KILL_PROCESS = 0x80000000
RET_ERRNO = 0x00050000
RET_ALLOW = 0x7FFF0000
# Offsets into struct seccomp_data.
SYSCALL_NUMBER, ARCHITECTURE, FIRST_ARGUMENT = 0, 4, 16  # low 32 bits of args[0]


def _instruction(code: int, k: int, jump_true: int = 0, jump_false: int = 0) -> bytes:
    return struct.pack("<HBBI", code, jump_true, jump_false, k)


def build_filter(denied: dict[str, int] = DENIED_SYSCALLS) -> bytes:
    """Return the filter as packed `struct sock_filter` entries."""
    program = [
        _instruction(LOAD_WORD, ARCHITECTURE),
        _instruction(JUMP_EQUAL, AUDIT_ARCH_X86_64, jump_true=1),
        _instruction(RETURN, RET_KILL_PROCESS),
        _instruction(LOAD_WORD, SYSCALL_NUMBER),
        _instruction(JUMP_AT_LEAST, X32_SYSCALL_BIT, jump_false=1),
        _instruction(RETURN, RET_KILL_PROCESS),
    ]
    # socket/socketpair: jump into the family check, otherwise skip past it.
    families = len(ALLOWED_SOCKET_FAMILIES)
    program += [
        _instruction(JUMP_EQUAL, SOCKET_CALLS[0], jump_true=1),
        _instruction(JUMP_EQUAL, SOCKET_CALLS[1], jump_false=families + 3),
        _instruction(LOAD_WORD, FIRST_ARGUMENT),
        *[_instruction(JUMP_EQUAL, family, jump_true=families - index)
          for index, family in enumerate(ALLOWED_SOCKET_FAMILIES)],
        _instruction(RETURN, RET_ERRNO | errno.EAFNOSUPPORT),
        _instruction(RETURN, RET_ALLOW),
    ]
    for number in sorted(denied.values()):
        program.append(_instruction(JUMP_EQUAL, number, jump_false=1))
        program.append(_instruction(RETURN, RET_ERRNO | errno.EPERM))
    program.append(_instruction(RETURN, RET_ALLOW))
    return b"".join(program)
