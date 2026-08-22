"""Make the in-tree python/ package importable when running from a dev
build (the native _fusedtok module is expected on PYTHONPATH, pointing at
the CMake build directory)."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYPKG = os.path.join(_ROOT, "python")
if _PYPKG not in sys.path:
    sys.path.insert(0, _PYPKG)
