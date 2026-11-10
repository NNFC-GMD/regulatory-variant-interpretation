"""Shared pieces of the ChromBPNet-against-Cherimoya comparison (compare.sh, levels 1-5).

Each level is one script in this directory, run in the cherimoya environment inside a box. A script works on units
(a dataset, or every dataset at once for a summary) and keeps one done marker per unit, so compare.sh can be run
again at any time: a unit whose marker still describes its inputs, settings and outputs is left alone, and one whose
inputs are not there yet exits WAITING (3) without writing anything, so compare.sh records it and goes on.

What lives here:
  - the roots (CHERI_ROOT, CHROMBPNET_ROOT, COMPENDIA_ROOT, COMPARISON_ROOT and the igvf folds directory) and the
    dataset lists;
  - done markers: the inputs' path, size and mtime (an optional input that is absent is recorded as absent, so its
    later arrival makes the unit run again), the settings, the outputs' size and mtime, and FORMAT, a number each
    script bumps when what it writes changes;
  - atomic writes (a <name>.partial.<pid> file renamed into place);
  - statistics shared by the levels: the percentile bootstrap over datasets and the two-sided Wilcoxon signed-rank
    test;
  - the plot style of plot_comparison.py: each dataset keeps one Okabe-Ito colour and marker (in sorted dataset
    order) in every figure, and names lose the prefix all of them share;
  - the chromosome role (train, valid, test) of each fold, from igvf_tf_chrombpnet's folds/fold_<k>.json;
  - narrowPeak reading and the join of two beds by peak name (never by row index).
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (*(str(HERE.parent / d) for d in ("train", "attribution", "downstream")), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_jobs  # noqa: E402
import plot_comparison  # noqa: E402

logger = logging.getLogger("compare")

MODELS = ("chrombpnet", "cherimoya")
NAMES = {"chrombpnet": "ChromBPNet", "cherimoya": "Cherimoya"}
WAITING = 3
"""Exit status of a unit whose inputs are not there yet: compare.sh records it and goes on."""
HEAD = "counts"
"""The head both models are compared on (the plan's contrib_heads)."""
ROLES = ("train", "valid", "test")
ROLE_KEYS = {"training_chroms": "train", "validation_chroms": "valid", "test_chroms": "test"}


class Waiting(Exception):
    """An input of the unit is not there yet; the unit exits WAITING."""


class Failed(Exception):
    """The inputs are there but wrong (a join that does not hold, a shape that does not fit); the unit exits 1."""


# ---------------------------------------------------------------------------------------------------------------------
# roots, lists, logging


def add_root_args(ap: argparse.ArgumentParser) -> None:
    """--cheri-root, --chrombpnet-root, --compendia-root, --comparison-root and --folds-dir, each defaulting to the
    environment compare.sh exports (CHERI_ROOT, CHROMBPNET_ROOT, COMPENDIA_ROOT, COMPARISON_ROOT, REPO_ROOT/folds; the
    folds default to the vendored copy without REPO_ROOT)."""
    env = os.environ.get
    repo = env("REPO_ROOT")
    ap.add_argument("--cheri-root", type=Path, default=env("CHERI_ROOT"), help="make_jobs.py's --out-root")
    ap.add_argument("--chrombpnet-root", type=Path, default=env("CHROMBPNET_ROOT"),
                    help="the ChromBPNet results root (<root>/<dataset>/...)")  # fmt: skip
    ap.add_argument("--compendia-root", type=Path, default=env("COMPENDIA_ROOT"),
                    help="09.0's compendia (<root>/<tag>/, <root>/inputs/)")  # fmt: skip
    ap.add_argument("--comparison-root", type=Path, default=env("COMPARISON_ROOT"), help="where the outputs go")
    vendored = Path(__file__).resolve().parents[2] / "vendor" / "igvf_tf_chrombpnet" / "folds"
    ap.add_argument("--folds-dir", type=Path, default=Path(repo) / "folds" if repo else vendored,
                    help="igvf_tf_chrombpnet's folds/ (fold_<k>.json); default $REPO_ROOT/folds, else this "
                         "repository's copy, vendor/igvf_tf_chrombpnet/folds")  # fmt: skip


def require(args: argparse.Namespace, *names: str) -> None:
    missing = [n for n in names if getattr(args, n) is None]
    if missing:
        raise SystemExit("set " + ", ".join("--" + n.replace("_", "-") for n in missing) + " (or the environment)")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%F %T")  # fmt: skip


def read_list(path: Path) -> list[str]:
    """A dataset list: one name per line, blank lines and # comments skipped, repeats dropped, order kept."""
    out: list[str] = []
    for line in Path(path).read_text().splitlines():
        name = line.split("#", 1)[0].strip()
        if name and name not in out:
            out.append(name)
    if not out:
        raise SystemExit(f"{path} names no dataset")
    return out


def result(note: str) -> None:
    """The unit's one-line outcome, the line compare.sh copies into its record."""
    print(f"RESULT: {note}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# done markers


def stat_entry(path: Path | str) -> dict:
    """{path, size, mtime_ns} of a file, or {path, absent: True}."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return {"path": str(path), "absent": True}
    return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def fingerprint(paths: Mapping[str, Path | str | None]) -> dict[str, dict]:
    return {role: stat_entry(p) for role, p in sorted(paths.items()) if p is not None}


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=20, check=False)  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


class Marker:
    """A unit's done marker: <dir>/.done/<unit>.json.

    current() is None when the marker records this FORMAT, these settings and these inputs (by path, size and
    mtime; absent ones as absent), and every output it records still has its size and mtime; otherwise it is the
    reason to run. write() stats the inputs again first and refuses (Failed) if one changed while the unit ran.
    """

    def __init__(self, directory: Path, unit: str, fmt: int, inputs: Mapping[str, Path | str | None],
                 settings: Mapping) -> None:  # fmt: skip
        self.path = Path(directory) / ".done" / f"{unit}.json"
        self.unit = unit
        self.fmt = fmt
        self.inputs = fingerprint(inputs)
        self.settings = json.loads(json.dumps(settings, default=str))

    def current(self) -> str | None:
        try:
            mark = json.loads(self.path.read_text())
        except FileNotFoundError:
            return "not done yet"
        except (OSError, ValueError) as exc:
            return f"unreadable {self.path}: {exc}"
        if mark.get("format") != self.fmt:
            return f"made by format {mark.get('format')}, this is {self.fmt}"
        if mark.get("settings") != self.settings:
            return "other settings"
        if mark.get("inputs") != self.inputs:
            changed = sorted(r for r in set(self.inputs) | set(mark.get("inputs") or {})
                             if (mark.get("inputs") or {}).get(r) != self.inputs.get(r))  # fmt: skip
            return "inputs changed: " + ", ".join(changed[:8]) + (" ..." if len(changed) > 8 else "")
        for name, out in (mark.get("outputs") or {}).items():
            if stat_entry(out.get("path", "")) != out:
                return f"output {name} changed or is gone"
        return None

    def write(self, outputs: Mapping[str, Path | str], note: str, extra: Mapping | None = None) -> None:
        now = fingerprint({role: v["path"] for role, v in self.inputs.items()})
        if now != self.inputs:
            changed = sorted(r for r in now if now[r] != self.inputs.get(r))
            raise Failed(f"inputs changed while {self.unit} ran: {', '.join(changed)}; run it again")
        mark = {
            "unit": self.unit,
            "format": self.fmt,
            "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "commit": git_commit(),
            "settings": self.settings,
            "inputs": self.inputs,
            "outputs": fingerprint(outputs),
            "note": note,
        }
        if extra:
            mark.update(extra)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.path, mark)

    def remove(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------------------------------------------------
# writing


def partial(path: Path) -> Path:
    return path.with_name(f"{path.name}.partial.{os.getpid()}")


def atomic(path: Path, write: Callable[[Path], None]) -> Path:
    """write(tmp) then rename tmp to path; tmp is removed if write fails."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = partial(path)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def write_json(path: Path, data) -> Path:
    return atomic(Path(path), lambda tmp: tmp.write_text(json.dumps(data, indent=2, default=str) + "\n"))


def write_text(path: Path, text: str) -> Path:
    return atomic(Path(path), lambda tmp: tmp.write_text(text))


def write_table(path: Path, df, float_format: str = "%.6g") -> Path:
    """A pandas DataFrame as TSV (gzip when the name ends in .gz, at level 4: fast and still small)."""
    path = Path(path)

    def write(tmp: Path) -> None:
        if path.name.endswith(".gz"):
            with gzip.open(tmp, "wt", compresslevel=4, newline="") as fh:
                df.to_csv(fh, sep="\t", index=False, float_format=float_format, na_rep="NA", lineterminator="\n")
        else:
            df.to_csv(tmp, sep="\t", index=False, float_format=float_format, na_rep="NA", lineterminator="\n")

    return atomic(path, write)


def save_figure(fig, out: Path) -> list[Path]:
    """<out>.pdf and <out>.png, each written atomically."""
    paths = []
    for ext in ("pdf", "png"):
        target = Path(f"{out}.{ext}")
        paths.append(atomic(target, lambda tmp, ext=ext: fig.savefig(tmp, format=ext, dpi=200, bbox_inches="tight")))
    return paths


# ---------------------------------------------------------------------------------------------------------------------
# statistics


def bootstrap_ci(values: Iterable[float], stat: Callable = np.median, n_boot: int = 10000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float, float]:  # fmt: skip
    """The statistic of the values and its percentile bootstrap interval (values resampled with replacement)."""
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    est = float(stat(x))
    if x.size == 1:
        return (est, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = x[rng.integers(0, x.size, size=(n_boot, x.size))]
    boot = np.apply_along_axis(stat, 1, draws) if stat not in (np.median, np.mean) else stat(draws, axis=1)
    lo, hi = np.quantile(boot, [(1 - level) / 2, 1 - (1 - level) / 2])
    return (est, float(lo), float(hi))


def wilcoxon(diffs: Iterable[float]) -> tuple[float, float, int]:
    """Two-sided Wilcoxon signed-rank test of paired differences: (statistic, p, n non-zero). NaN when fewer than one
    non-zero difference (scipy's exact distribution for small n, its normal approximation otherwise)."""
    from scipy.stats import wilcoxon as _wilcoxon

    d = np.asarray([v for v in diffs if np.isfinite(v)], dtype=np.float64)
    n = int(np.count_nonzero(d))
    if n == 0:
        return (np.nan, np.nan, 0)
    res = _wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return (float(res.statistic), float(res.pvalue), n)


# ---------------------------------------------------------------------------------------------------------------------
# plot style (plot_comparison.py's)


def dataset_styles(datasets: Iterable[str]) -> dict[str, tuple[str, str]]:
    """{dataset: (colour, marker)}: plot_comparison.style() in sorted dataset order, as plot_comparison.py assigns them,
    so a dataset looks the same in every figure."""
    return {ds: plot_comparison.style(i) for i, ds in enumerate(sorted(set(datasets)))}


def short_names(datasets: Iterable[str]) -> dict[str, str]:
    """{dataset: label}: the names without the prefix (up to an underscore) that all of them share."""
    names = sorted(set(datasets))
    prefix = os.path.commonprefix(names) if len(names) > 1 else ""
    prefix = prefix[: prefix.rfind("_") + 1] if "_" in prefix else ""
    return {ds: ds[len(prefix) :] or ds for ds in names}


def matplotlib_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#DDDDDD", "grid.linewidth": 0.4, "font.size": 8})  # fmt: skip
    return plt


def dataset_legend(fig, styles: Mapping[str, tuple[str, str]], labels: Mapping[str, str], title: str) -> None:
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], ls="", marker=m, color=c, markeredgecolor="#333333", markeredgewidth=0.4,
                      markersize=6, label=labels[ds]) for ds, (c, m) in styles.items()]  # fmt: skip
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=7,
               title=title, title_fontsize=7, ncol=1 if len(handles) <= 20 else 2)  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------------
# folds and peaks


def fold_roles(folds_dir: Path, folds: Iterable[str]) -> dict[str, dict[str, str]]:
    """{fold: {chrom: train|valid|test}} from igvf_tf_chrombpnet's folds/fold_<k>.json (make_jobs.read_fold's checks:
    no chromosome in two lists)."""
    if folds_dir is None:
        raise Failed("no folds directory (--folds-dir or REPO_ROOT)")
    out = {}
    for k in folds:
        try:
            split = make_jobs.read_fold(Path(folds_dir), str(k))
        except (OSError, ValueError) as exc:
            raise Failed(f"cannot read the split of fold {k} in {folds_dir}: {exc}") from exc
        out[str(k)] = {c: ROLE_KEYS[key] for key, chroms in split.items() for c in chroms}
    return out


def heldout_fold(roles: Mapping[str, Mapping[str, str]], chrom: str) -> str | None:
    """The fold whose test set holds the chromosome (None if no fold tests it)."""
    tested = [k for k, r in roles.items() if r.get(chrom) == "test"]
    return tested[0] if len(tested) == 1 else None


NARROWPEAK = ("chrom", "start", "end", "name", "score", "strand", "signal", "pvalue", "qvalue", "summit")


def read_bed(path: Path):
    """A narrowPeak (10 columns, peak names in column 4) as a DataFrame, rows in file order. Short beds (3+ columns)
    get the name chrom:start-end and the summit at the midpoint."""
    import pandas as pd

    df = pd.read_csv(path, sep="\t", header=None, comment=None, dtype=str, keep_default_na=False, quoting=3)
    if df.shape[1] < 3:
        raise Failed(f"{path}: fewer than 3 columns")
    df = df.iloc[:, : len(NARROWPEAK)]
    df.columns = list(NARROWPEAK[: df.shape[1]])
    df["start"] = df["start"].astype(np.int64)
    df["end"] = df["end"].astype(np.int64)
    if "name" not in df:
        df["name"] = df["chrom"] + ":" + df["start"].astype(str) + "-" + df["end"].astype(str)
    if "summit" not in df:
        df["summit"] = (df["end"] - df["start"]) // 2
    df["summit"] = df["summit"].astype(np.int64)
    df["signal"] = pd.to_numeric(df["signal"], errors="coerce") if "signal" in df else np.nan
    return df


def join_by_name(left, right, what: str = "beds") -> tuple[np.ndarray, np.ndarray]:
    """(li, ri): the rows of two beds with the same peak name, in the left bed's order, after checking that names are
    unique in each and that joined rows sit at the same chrom, start, end and summit."""
    import pandas as pd

    for side, df in (("left", left), ("right", right)):
        dup = df["name"].duplicated()
        if dup.any():
            raise Failed(f"{what}: {int(dup.sum())} repeated peak names in the {side} bed; the join needs unique names")
    pos = pd.Series(np.arange(len(right)), index=right["name"].to_numpy())
    ri = pos.reindex(left["name"].to_numpy()).to_numpy()
    keep = ~np.isnan(ri.astype(np.float64))
    li = np.flatnonzero(keep)
    ri = ri[keep].astype(np.int64)
    for col in ("chrom", "start", "end", "summit"):
        if not np.array_equal(left[col].to_numpy()[li], right[col].to_numpy()[ri]):
            bad = int((left[col].to_numpy()[li] != right[col].to_numpy()[ri]).sum())
            raise Failed(f"{what}: {bad} rows joined by name differ in {col}")
    return li, ri
