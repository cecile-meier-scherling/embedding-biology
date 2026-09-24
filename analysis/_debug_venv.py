"""Probe why `uv run jupyter lab` cannot find python3. Remove after debug."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

LOG = Path(
    "/Users/cmeiersc/Brown Dropbox/Cecile Meier-Scherling/Brown/git/axiom_takehome/.cursor/debug-4767a1.log"
)
ROOT = Path(__file__).resolve().parents[1]


def log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    LOG.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sessionId": "4767a1",
        "runId": os.environ.get("DEBUG_RUN_ID", "pre-fix"),
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with LOG.open("a") as f:
        f.write(json.dumps(payload) + "\n")
    # #endregion


def main() -> None:
    jupyter = ROOT / ".venv/bin/jupyter"
    current_python = ROOT / ".venv/bin/python3"
    old_python = ROOT / "axiom/.venv/bin/python3"
    shebang_lines = jupyter.read_text().splitlines()[:3] if jupyter.exists() else []
    shebang_target = ""
    for line in shebang_lines:
        if "exec" in line and ".venv" in line:
            quoted = [part for part in line.split("'") if ".venv" in part]
            shebang_target = quoted[0] if quoted else line

    # A: jupyter shebang still points at axiom/.venv
    log(
        "A",
        "analysis/_debug_venv.py:shebang",
        "jupyter shebang target",
        {
            "jupyter_exists": jupyter.exists(),
            "shebang_lines": shebang_lines,
            "shebang_target": shebang_target,
            "points_at_old_axiom_venv": "axiom/.venv" in shebang_target,
        },
    )

    # B: old interpreter path is gone
    log(
        "B",
        "analysis/_debug_venv.py:old_python",
        "old axiom venv python",
        {
            "old_python": str(old_python),
            "old_python_exists": old_python.exists(),
            "axiom_dir_exists": (ROOT / "axiom").exists(),
        },
    )

    # C: current venv python works
    log(
        "C",
        "analysis/_debug_venv.py:current_python",
        "current root venv python",
        {
            "current_python": str(current_python),
            "current_python_exists": current_python.exists(),
            "current_python_is_file": current_python.is_file(),
        },
    )

    # D: Dropbox path aliases
    brown = Path("/Users/cmeiersc/Brown Dropbox/Cecile Meier-Scherling/Brown/git/axiom_takehome")
    dropbox = Path("/Users/cmeiersc/Dropbox (Brown)/Brown/git/axiom_takehome")
    log(
        "D",
        "analysis/_debug_venv.py:paths",
        "dropbox path comparison",
        {
            "cwd": os.getcwd(),
            "root": str(ROOT),
            "brown_exists": brown.exists(),
            "dropbox_exists": dropbox.exists(),
            "same_inode": brown.exists()
            and dropbox.exists()
            and brown.stat().st_ino == dropbox.stat().st_ino,
        },
    )

    # E: uv project location
    log(
        "E",
        "analysis/_debug_venv.py:project",
        "uv project files",
        {
            "root_pyproject": (ROOT / "pyproject.toml").exists(),
            "axiom_pyproject": (ROOT / "axiom/pyproject.toml").exists(),
            "project_name": "axiom" if (ROOT / "pyproject.toml").exists() else None,
        },
    )
    print(f"wrote debug logs to {LOG}")


if __name__ == "__main__":
    main()
