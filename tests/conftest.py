"""Make the in-tree python/ package importable when running from a dev
build (the native _fusedtok module is expected on PYTHONPATH, pointing at
the CMake build directory).

An INSTALLED fusedtok (pip wheel) wins over the source tree: the source
package alone has no native module, so prepending it would shadow the
wheel and break every import. Only prepend when fusedtok is not already
importable - which is exactly the dev layout, where PYTHONPATH carries
the build directory with the compiled extension."""

import importlib
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYPKG = os.path.join(_ROOT, "python")

try:
    importlib.import_module("fusedtok")
except ImportError:
    if _PYPKG not in sys.path:
        sys.path.insert(0, _PYPKG)
