"""Read ``config/<dataset>/config.yaml`` and export it to shell or Python.

One config file per dataset is the whole configuration. There is no second
"site" file: the handful of machine-specific values (conda paths, where the
references live) are environment variables with defaults, because they are the
same for every dataset and should never be committed.

**This module must run on the system python, with no third-party imports and no
conda environment active** -- ``lib/bash/config.sh`` calls it to turn the YAML
into shell assignments *before* a step activates anything. Hence: no
``utils.log`` import, no PyYAML requirement, nothing from the rest of the
package. It is executable directly::

    python3 lib/python/utils/config.py export config/igvf3_cardiomyocyte/config.yaml
    python3 lib/python/utils/config.py json   config/igvf3_cardiomyocyte/config.yaml

Parsing
-------
PyYAML is used when it is importable (the conda envs have it) and a strict
built-in parser otherwise. The built-in parser handles exactly what these
configs use -- scalars, flow and block lists, and one level of nested mapping --
and **raises on anything else** rather than guessing. That is the important
property: a silently misparsed config would be far worse than a loud failure.
``tests/test_config.py`` asserts the two parsers agree on every config in the
repo, so the fallback cannot drift from real YAML unnoticed.

Values are emitted for bash with ``${...}`` left intact, so the shell still does
the interpolation it always did (``fragments_path: "${dataset_dir}/data"``).
"""

# NO `from __future__ import annotations` here, deliberately. This module is
# the first thing lib/bash/config.sh runs, before any conda env exists, and on
# Sherlock that interpreter is /usr/bin/python3 = 3.6.8 -- which rejects that
# import outright ("future feature annotations is not defined", it arrived in
# 3.7) and so could not even parse this file. tests/test_config.py's
# test_loader_runs_on_bare_system_python pins the contract.
#
# Nothing here needs it: every function signature uses plain str/dict/int, and
# the one modern annotation (`container: str | None` in _parse_builtin) is a
# LOCAL variable annotation, which Python never evaluates at runtime.

import json
import re
import shlex
import sys
from pathlib import Path

__all__ = ["ConfigError", "load", "parse", "resolve", "signal_type_for", "to_shell"]


class ConfigError(ValueError):
    """A config that cannot be parsed, or uses YAML this loader does not support."""


# Constructs the built-in parser deliberately refuses rather than mis-reading.
_UNSUPPORTED = {
    "&": "anchors",
    "*": "aliases",
    "|": "literal block scalars",
    ">": "folded block scalars",
    "!": "tags",
}


def _scalar(text: str):
    """YAML scalar -> Python. Strings stay strings; only unambiguous forms convert."""
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    low = text.lower()
    if low in {"true", "yes"}:
        return True
    if low in {"false", "no"}:
        return False
    if low in {"null", "~", ""}:
        return None
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d*\.\d+", text):
        return float(text)
    return text


def _check_supported(line: str, lineno: int) -> None:
    stripped = line.split("#", 1)[0].strip()
    if not stripped:
        return
    _, _, value = stripped.partition(":")
    value = value.strip()
    if value and value[0] in _UNSUPPORTED:
        raise ConfigError(
            f"line {lineno}: {_UNSUPPORTED[value[0]]} are not supported by the "
            f"built-in parser.\n  {line.strip()}\n"
            "Install PyYAML in the environment reading this config, or rewrite "
            "the value as a plain scalar or list."
        )


def _parse_builtin(text: str) -> dict:
    """Strict subset parser: scalars, flow/block lists, one nested mapping level.

    A key with an empty value opens a container whose type is decided by its
    first child: ``- item`` makes a list, ``k: v`` makes a mapping.
    """
    root: dict = {}
    container: str | None = None

    for lineno, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        _check_supported(raw, lineno)
        line = raw.split(" #", 1)[0].rstrip()
        indent = len(line) - len(line.lstrip())
        body = line.strip()

        if indent == 0:
            container = None
            if ":" not in body:
                raise ConfigError(f"line {lineno}: expected 'key: value'\n  {body}")
            key, _, value = body.partition(":")
            key, value = key.strip().strip("\"'"), value.strip()
            if value == "":
                root[key] = None  # container; type set by the first child
                container = key
            elif value.startswith("[") and value.endswith("]"):
                inner = value[1:-1].strip()
                root[key] = [_scalar(v) for v in inner.split(",")] if inner else []
            else:
                root[key] = _scalar(value)
            continue

        if container is None:
            raise ConfigError(f"line {lineno}: unexpected indentation\n  {body}")

        if body.startswith("- "):
            if root[container] is None:
                root[container] = []
            if not isinstance(root[container], list):
                raise ConfigError(f"line {lineno}: list item inside a mapping\n  {body}")
            root[container].append(_scalar(body[2:]))
            continue

        if ":" not in body:
            raise ConfigError(f"line {lineno}: expected 'key: value'\n  {body}")
        key, _, value = body.partition(":")
        if value.strip() == "":
            raise ConfigError(f"line {lineno}: only one level of nesting is supported\n  {body}")
        if root[container] is None:
            root[container] = {}
        if not isinstance(root[container], dict):
            raise ConfigError(f"line {lineno}: mapping entry inside a list\n  {body}")
        root[container][key.strip().strip("\"'")] = _scalar(value)

    return root


def parse(text: str) -> dict:
    """Parse YAML text. Uses PyYAML when available, the strict subset otherwise."""
    try:
        import yaml  # noqa: PLC0415  # optional: absent on the bootstrap python
    except ImportError:
        return _parse_builtin(text)
    loaded = yaml.safe_load(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError("config must be a mapping at the top level")
    return loaded


def load(path) -> dict:
    """Read and parse a config file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"no config file at {path}")
    try:
        return resolve(parse(path.read_text()))
    except ConfigError:
        raise
    except Exception as exc:  # a PyYAML error
        raise ConfigError(f"{path}: {exc}") from exc


# Signal kind by filename suffix, longest match first so ".fragments.tsv.gz"
# beats ".tsv.gz". One table, applied once in resolve(), so bash and Python
# cannot disagree about what a file is.
SIGNAL_SUFFIXES = [
    (".fragments.tsv.gz", "fragments"),
    (".fragments.tsv", "fragments"),
    (".tagalign.gz", "tagalign"),
    (".tagalign", "tagalign"),
    (".tsv.gz", "fragments"),
    (".tsv", "fragments"),
    (".bedpe.gz", "fragments"),
    (".bam", "bam"),
    (".bigwig", "bigwig"),
    (".bw", "bigwig"),
]

#: chrombpnet trains from these (-ifrag / -itag / -ibam).
TRAINABLE_SIGNALS = {"fragments", "tagalign", "bam"}


def signal_type_for(path: str) -> str:
    """Infer the signal kind from a filename.

    ``bigwig`` means the signal is already a bigwig: 00.0 registers it as the
    prepared bigwig, which training hands to chrombpnet with -bw, so the reads
    conversion is skipped entirely -- but it must already carry the Tn5 shift
    chrombpnet expects.
    """
    name = str(path).strip().lower()
    for suffix, kind in SIGNAL_SUFFIXES:
        if name.endswith(suffix):
            return kind
    known = ", ".join(sorted({s for s, _ in SIGNAL_SUFFIXES}))
    raise ConfigError(
        f"cannot tell the signal type from {path!r}.\n"
        f"  Recognised suffixes: {known}\n"
        "  Rename the file, or symlink it to a recognised extension."
    )


def resolve(config: dict) -> dict:
    """Apply defaults and derivations. The single place either language does this."""
    out = dict(config)

    name = out.get("dataset_name")
    if name:
        # The step loops still iterate a `datasets` array; one name means one entry.
        out.setdefault("datasets", [name])
        out.setdefault("bias_dataset", name)

    out.setdefault("assay", "ATAC")
    out.setdefault("peak_type", "all")

    # GC-matched negatives (01.0). These mirror `chrombpnet prep nonpeaks`
    # defaults; they live here so a dataset can override them in config.yaml
    # and so every run records the values it actually used -- the negatives are
    # training data, and the seed is what makes them reproducible.
    out.setdefault("nonpeak_seed", 1234)
    out.setdefault("neg_to_pos_ratio", 2)
    out.setdefault("nonpeak_stride", 1000)

    # QC (02.0) compares peaks to negatives over one fixed window. 1000 is
    # ChromBPNet's OUTPUT window -- the span its counts head predicts -- so the
    # separation measured is the one the model gets scored on.
    out.setdefault("qc_compare_window", 1000)

    if out.get("signal_path"):
        out["signal_type"] = signal_type_for(out["signal_path"])
        out["signal_is_prepared_bigwig"] = out["signal_type"] == "bigwig"

    if out.get("folds") and not out.get("bias_sweep_folds"):
        out["bias_sweep_folds"] = list(out["folds"])

    return out


def _shell_value(value) -> str:
    """Quote for bash, but leave ${...} for the shell to expand."""
    text = (
        ""
        if value is None
        else ("true" if value is True else "false" if value is False else str(value))
    )
    return f'"{text}"' if "${" in text else shlex.quote(text)


def to_shell(config: dict) -> str:
    """Render a config as bash assignments: scalars, arrays and associative arrays."""
    lines = []
    for key, value in config.items():
        if not re.fullmatch(r"[A-Za-z_]\w*", key):
            raise ConfigError(f"{key!r} is not usable as a shell variable name")
        if isinstance(value, dict):
            pairs = " ".join(f"[{k}]={_shell_value(v)}" for k, v in value.items())
            lines.append(f"declare -A {key}=( {pairs} )")
        elif isinstance(value, list):
            lines.append(f"{key}=( {' '.join(_shell_value(v) for v in value)} )")
        else:
            lines.append(f"{key}={_shell_value(value)}")
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2 or argv[0] not in {"export", "json"}:
        print(f"usage: {Path(__file__).name} {{export|json}} <config.yaml>", file=sys.stderr)
        return 2
    mode, path = argv
    try:
        config = load(path)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(to_shell(config) if mode == "export" else json.dumps(config, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
