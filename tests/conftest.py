import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

_LIB = _ROOT / "dist" / "libmojo-minhash.so"

if not _LIB.exists():
    pytest.skip(
        "libmojo-minhash.so not built; run `bash build/build.sh`",
        allow_module_level=True,
    )

# Fixtures used by more than one test module live in parity_helpers and are
# registered here, where pytest always discovers them.
from parity_helpers import coeffs_json_cleanup  # noqa: E402,F401
