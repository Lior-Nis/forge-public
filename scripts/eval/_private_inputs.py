"""Resolve inputs that live in the author-private `research/` tree.

`research/` is gitignored (see .gitignore) — it holds paper drafts and the
derived prediction vectors behind several supplementary analyses. The scripts
that read it therefore CANNOT run in a clone of this repository, and previously
failed with a bare FileNotFoundError that gave no hint why.

Point `FORGE_PRIVATE_DATA` at the tree if you have it; otherwise these scripts
report themselves as author-only and exit cleanly.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ROOT = "research/paper_final/data"
ENV_VAR = "FORGE_PRIVATE_DATA"


def private_path(*parts: str, output: bool = False) -> Path:
    """Return a path under the private data root, or exit with an explanation.

    Fails at the same point the old code did (module import), but says what is
    missing, that it is expected to be missing in a public clone, and how an
    author with the tree can point at it.

    Pass `output=True` for a path this script WRITES: only its parent directory
    has to exist.
    """
    root = Path(os.environ.get(ENV_VAR, DEFAULT_ROOT))
    target = root.joinpath(*parts)
    # An output (a cache the script writes, a results CSV) legitimately does not
    # exist yet; requiring it would stop an author WITH the tree from ever
    # creating it. For those, the containing directory is what must be present.
    if (target.parent if output else target).exists():
        return target
    raise SystemExit(
        f"[author-only] This script needs {target}, which is not in this "
        f"repository.\n"
        f"  `{DEFAULT_ROOT}` is gitignored private research material, so the "
        f"supplementary analysis this script performs cannot be reproduced from "
        f"a public clone.\n"
        f"  If you have the tree, set {ENV_VAR}=/path/to/paper_final/data.\n"
        f"  The headline results do NOT depend on this script — see REPRODUCE.md."
    )
