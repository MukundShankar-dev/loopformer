"""Skip marker for tests that start real sandboxes."""
import os
import platform
import shutil

import pytest

needs_sandbox = pytest.mark.skipif(
    platform.system() != "Linux" or platform.machine() != "x86_64" or shutil.which("bwrap") is None
    or "DBUS_SESSION_BUS_ADDRESS" not in os.environ,
    reason="needs Linux x86_64 with bubblewrap and a systemd user session")
