"""Read and check a motif compendium's input list: one ``label<TAB>path`` row per input.

Two readers share these rules, so 09.0 and the tool it runs cannot disagree about a list:

- ``workflows/SLURM/09.0.cross_dataset_compendium.sh`` in list mode (``COMPENDIUM_INPUTS``). A row's
  path is a dataset config (``config.yaml``), which 09.0 resolves to the MoDISco H5 that 08.0 writes
  for it, or a MoDISco results H5, used as it is.
- ``src/motif_compendium.py``, on the resolved ``label<TAB>H5`` table (``modisco_config.tsv``) it
  builds the compendium from. There every path must be an H5.

The rules:

- Blank lines and lines whose first non-blank character is ``#`` are skipped.
- Every other line is exactly two tab-separated fields, ``label`` and ``path``.
- A label matches ``^[A-Za-z0-9_]+$``. MotifCompendium names each motif
  ``<label>-<pos|neg>.pattern_<n>`` and splits those names on ``-`` and ``.``, and its export rejects
  ``/``.
- Labels are unique, and so are paths. A table keyed by label would keep only the last of two rows
  with one label, and one H5 under two labels would enter the compendium twice.
- A path ending in ``.yaml`` or ``.yml`` is a dataset config, one ending in ``.h5`` or ``.hdf5`` a
  MoDISco results file. Anything else is an error. A relative path is relative to the current
  directory, as on a command line.
- A list with no rows is an error.

Every problem in a list is reported at once, as ``preflight_check`` does for missing inputs.
Whether a path exists is a separate question (``split_missing``): 09.0 resolves configs first, and a
missing H5 is an error or a warning depending on the caller.

stdlib only, and nothing else from ``utils``: 09.0 runs this file as a script on ``bootstrap_python``
before any environment is active (lib/bash/common.sh), as config.sh runs ``utils/config.py``::

    python3 lib/python/utils/compendium_inputs.py parse <list.tsv>

prints one ``label<TAB>kind<TAB>absolute path`` line per row (kind is ``config`` or ``h5``), or
every problem on stderr and exit status 1.
"""

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "KINDS",
    "LABEL_PATTERN",
    "InputListError",
    "Row",
    "kind_of",
    "parse",
    "read",
    "split_missing",
]

#: What a label may contain: MotifCompendium's motif names are ``<label>-<pos|neg>.pattern_<n>``.
LABEL_PATTERN = r"[A-Za-z0-9_]+"

#: Path suffix -> what the row names.
KINDS = {".yaml": "config", ".yml": "config", ".h5": "h5", ".hdf5": "h5"}


class InputListError(ValueError):
    """A list with one or more problems; ``problems`` holds every one of them."""

    def __init__(self, source: str, problems: list):
        self.source = source
        self.problems = list(problems)
        lines = "\n".join(f"  {p}" for p in self.problems)
        super().__init__(f"{source}: {len(self.problems)} problem(s):\n{lines}")


@dataclass(frozen=True)
class Row:
    """One input: its label, its path as written, what the path names, and its line in the list."""

    label: str
    path: str
    kind: str
    line: int


def kind_of(path: str):
    """``config`` or ``h5`` from the path's suffix, or None for anything else."""
    return KINDS.get(Path(path).suffix.lower())


def _same_file_key(path: str) -> str:
    return os.path.normpath(os.path.abspath(path))


def parse(text: str, source: str = "<input list>", kinds=("config", "h5")) -> list:
    """The rows of a list, in order. Raises InputListError naming every problem.

    ``kinds`` is what a row may name: ``("h5",)`` for a table that must be resolved already.
    """
    rows = []
    problems = []
    first_line_of_label = {}
    first_line_of_path = {}
    for lineno, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        fields = raw.split("\t")
        if len(fields) != 2:
            problems.append(
                f"line {lineno}: expected 'label<TAB>path', found {len(fields)} tab-separated "
                f"field(s): {raw.strip()!r}"
            )
            continue
        label, path = (f.strip() for f in fields)
        if not re.fullmatch(LABEL_PATTERN, label):
            problems.append(
                f"line {lineno}: label {label!r} may contain only letters, digits and '_' "
                "(MotifCompendium names motifs <label>-<pos|neg>.pattern_<n>)"
            )
        elif label in first_line_of_label:
            problems.append(
                f"line {lineno}: label {label!r} is already used on line {first_line_of_label[label]}"
            )
        else:
            first_line_of_label[label] = lineno
        if not path:
            problems.append(f"line {lineno}: no path for label {label!r}")
            continue
        kind = kind_of(path)
        if kind is None:
            wanted = " or ".join(
                {"config": "a dataset config (.yaml)", "h5": "a MoDISco H5 (.h5)"}[k] for k in kinds
            )
            problems.append(f"line {lineno}: {path!r} is not {wanted}")
            continue
        if kind not in kinds:
            problems.append(
                f"line {lineno}: {path!r} is a {kind}; this list takes {', '.join(kinds)} only"
            )
            continue
        key = _same_file_key(path)
        if key in first_line_of_path:
            problems.append(
                f"line {lineno}: {path!r} is already listed on line {first_line_of_path[key]}"
            )
            continue
        first_line_of_path[key] = lineno
        rows.append(Row(label=label, path=path, kind=kind, line=lineno))
    if not rows and not problems:
        problems.append("no inputs: every line is blank or a comment")
    if problems:
        raise InputListError(source, problems)
    return rows


def read(path, kinds=("config", "h5")) -> list:
    """Parse the list file at ``path``."""
    path = Path(path)
    if not path.is_file():
        raise InputListError(str(path), ["no such file"])
    return parse(path.read_text(), source=str(path), kinds=kinds)


def split_missing(rows) -> tuple:
    """(rows whose path exists, rows whose path does not), each in list order."""
    present = [r for r in rows if os.path.exists(r.path)]
    missing = [r for r in rows if not os.path.exists(r.path)]
    return present, missing


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2 or argv[0] != "parse":
        print(f"usage: {Path(__file__).name} parse <list.tsv>", file=sys.stderr)
        return 2
    try:
        rows = read(argv[1])
    except InputListError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for row in rows:
        print(f"{row.label}\t{row.kind}\t{os.path.abspath(row.path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
