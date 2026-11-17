"""Per-step run metadata, written as one JSON document per step invocation.

Every step emits a companion record describing what it read, what it wrote,
with which checksums, under which code and which tool versions. Many runs can
then be loaded and queried together::

    SELECT step, dataset, status, duration_s
    FROM read_json_auto('<root>/*/results/metadata/**/*.json', union_by_name=true)
    ORDER BY started_at DESC;

Schema notes, all of them chosen so that heterogeneous steps union cleanly in
DuckDB rather than producing a ragged table:

- Top level is flat scalars plus a few ``LIST<STRUCT>`` columns. Scalars stay
  queryable without unnesting.
- ``params`` is a list of ``{key, value}`` with ``value`` always a **string**.
  A plain object would give every step a different STRUCT and the union would
  be a mess. ``command`` keeps the full argv as ground truth; ``params`` is the
  curated subset worth querying.
- ``inputs`` and ``outputs`` share one shape, so they can be UNIONed into a
  single "files" view.
- Who ran it is recorded at two levels, both as flat scalars: ``user``/``uid``
  is the cluster account, ``git_user_name``/``git_user_email`` is the person
  from git config. They differ whenever a lab account is shared, and the git
  identity is the one that survives an opaque account name.

Checksums
---------
md5 is on by default and streamed in 8 MiB chunks. The expensive case is a
multi-GB fragments file, and that is precisely the step whose runtime already
dwarfs the hash. Set ``METADATA_CHECKSUMS=0`` to skip; the record then
carries ``md5: null`` with ``md5_skipped: "disabled"``, which is distinguishable
from a file that was simply absent.

Granularity
-----------
A numbered step name (``00.1.preprocess_peaks``) is one *sbatch job*, written by
the EXIT trap in lib/bash/common.sh. An unnumbered one (``preprocess_peaks``) is
one *tool invocation*, written by the Python script itself -- a job that loops
over datasets produces one job record and several tool records. The names are
kept distinct on purpose so the two granularities coexist in the same table
rather than double-counting; filter on ``step LIKE '__.%'`` for jobs only.

Failure
-------
Outputs are declared up front but hashed on exit, so a run that dies still
records what it *intended* to produce, with ``exists: false``. The record is
written for failed runs too (``status: "failed"`` plus ``error``). A step killed
with SIGKILL (SLURM OOM or hard preemption) cannot write anything -- the absence
of a record is itself the signal.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

from utils import log

logger = log.get_logger(__name__)

# 2: ENCODE/IGVF-style field names. md5->md5sum, size_bytes->file_size,
#    path->filepath (plus a `file` basename; NOT ENCODE's
#    submitted_file_name -- a bare name is what you read in a
#    query), tools->software_versions, run_id->uuid,
#    started_at->date_created, ended_at->date_completed,
#    status->run_status, inputs->derived_from,
#    outputs->output_files.
#    `status` is deliberately NOT reused for ok/failed: on the portal it means
#    an object's lifecycle (released / in progress / archived), not whether a
#    process exited 0, so run outcome lives in `run_status`. No @id, @type or
#    accession is emitted -- those are portal-assigned, and inventing them
#    would make a local record look like a registered IGVF object.
#    Naming rule: every field is unambiguous ON ITS OWN, because UNNEST
#    flattens these structs into one table where a bare `key`, `value`,
#    `name` or `path` says nothing. Hence parameter_name/parameter_value,
#    software_name/software_version, file_role/file/filepath/file_size.
SCHEMA_VERSION = 2

CHECKSUM_ENV_VAR = "METADATA_CHECKSUMS"
CHUNK_BYTES = 8 << 20  # 8 MiB

#: Packages worth recording when present. Absent ones are skipped silently --
#: each environment has a different subset. The jax CUDA plugin's version and
#: major say which GPU build ran; modisco is TF-MoDISco 2.5 (tfmodisco-lite).
TRACKED_DISTRIBUTIONS = [
    "chrombpnet",
    "duckdb",
    "finemo",
    "h5py",
    "hdf5plugin",
    "jax",
    "jax-cuda12-plugin",
    "jax-cuda13-plugin",
    "jaxlib",
    "keras",
    "matplotlib",
    "memelite",
    "modisco",
    "MotifCompendium",
    "numba",
    "numpy",
    "pandas",
    "polars",
    "pyranges1",
    "pysam",
    "ruranges",
    "scipy",
    "torch",
]


def repo_root() -> Path:
    """The repo root, located relative to this file."""
    return Path(__file__).resolve().parents[3]


def checksums_enabled() -> bool:
    return os.environ.get(CHECKSUM_ENV_VAR, "1").strip().lower() not in {"0", "false", "no"}


def md5sum(path, chunk_bytes: int = CHUNK_BYTES) -> str:
    """Streaming md5, identical to `md5sum(1)`. Constant memory."""
    digest = hashlib.md5()  # noqa: S324  # provenance fingerprint, not a security control
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


#: Multi-part extensions that mean something as a unit. Order matters: the
#: longest match wins, so "d0.fragments.tsv.gz" is tsv.gz, not gz.
COMPOUND_SUFFIXES = (
    ".bed.gz",
    ".tsv.gz",
    ".txt.gz",
    ".vcf.gz",
    ".fa.gz",
    ".fasta.gz",
    ".narrowPeak.gz",
    ".bedGraph.gz",
    ".gtf.gz",
)

#: Extension -> the name people actually use for the format.
FORMAT_ALIASES = {
    ".bw": "bigwig",
    ".bigwig": "bigwig",
    ".bigWig": "bigwig",
    ".fa": "fasta",
    ".fna": "fasta",
    ".h5": "h5",
    ".hdf5": "h5",
    ".tbi": "tbi",
    ".fai": "fai",
    ".narrowPeak": "narrowPeak",
}


def _peak_rss_gb() -> float | None:
    """Peak resident memory of this process and its finished children, in GiB.

    Never raises: provenance must not fail a step. Returns None where the
    platform does not provide it.
    """
    try:
        import resource

        scale = 1 if sys.platform == "darwin" else 1024  # macOS reports bytes
        peak = max(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        )
        return round(peak * scale / (1024**3), 3)
    except Exception:  # pragma: no cover - platform dependent
        return None


def file_format(path) -> str | None:
    """The ENCODE-style file_format: how the bytes are encoded, nothing else.

    Derived from the extension, never hand-written, because hand-written
    formats drift into the semantic name -- the old vocabulary had roles like
    "qc_tsv", "counts_h5" and "prepared_bigwig" that answered "what is this"
    and "how is it encoded" in one string. Those are separate questions:
    `parameter_name` answers the first, this answers the second.
    """
    p = Path(path)
    if p.is_dir():
        return "directory"
    name = p.name
    for suf in COMPOUND_SUFFIXES:
        if name.endswith(suf):
            return suf.lstrip(".")
    suffix = p.suffix
    if not suffix:
        return None
    return FORMAT_ALIASES.get(suffix, suffix.lstrip(".").lower())


def file_record(
    role: str, path, checksums: bool | None = None, parameter_type: str = "input"
) -> dict:
    """Describe one input or output file. Never raises."""
    if checksums is None:
        checksums = checksums_enabled()
    p = Path(path)
    resolved = str(p.resolve()) if p.exists() else str(p)
    rec = {
        # WHAT the data is -- signal, fragments, peaks, contributions, genome.
        # Never how it is encoded; that is file_format.
        "parameter_name": role,
        "parameter_value": resolved,
        "parameter_type": parameter_type,
        "file": p.name,
        "filepath": resolved,
        "file_format": file_format(p),
        "file_exists": p.exists(),
        "file_size": None,
        "file_mtime": None,
        "md5sum": None,
        "md5_skipped": None,
    }
    if not p.exists() or not p.is_file():
        rec["md5_skipped"] = "missing" if not p.exists() else "not_a_file"
        return rec
    st = p.stat()
    rec["file_size"] = st.st_size
    rec["file_mtime"] = _utc(st.st_mtime)
    if not checksums:
        rec["md5_skipped"] = "disabled"
        return rec
    try:
        rec["md5sum"] = md5sum(p)
    except OSError as exc:
        rec["md5_skipped"] = f"error: {exc}"
    return rec


def normalize_remote_url(url: str | None) -> str | None:
    """Turn any GitHub remote form into a browsable https URL.

    Handles ``git@github.com:Owner/Repo.git``, ``ssh://git@github.com/Owner/Repo``
    and ``https://github.com/Owner/Repo.git``. Returns None for anything it does
    not recognise rather than guessing.
    """
    if not url:
        return None
    url = url.strip().removesuffix(".git")
    if url.startswith("git@"):  # git@host:owner/repo
        host, _, path = url[4:].partition(":")
        return f"https://{host}/{path}" if path else None
    if url.startswith("ssh://"):
        rest = url[len("ssh://") :]
        rest = rest.partition("@")[2] or rest
        return f"https://{rest}"
    if url.startswith(("https://", "http://")):
        return url
    return None


def git_info() -> dict:
    """Commit, branch, dirty flag and browsable GitHub URLs.

    All values are None if git or .git is unavailable (e.g. the repo was copied
    to the cluster as a tarball), which is why every probe is wrapped.

    ``commit_url`` and ``script_url`` are permalinks pinned to the commit that
    was checked out, so a record points at exactly the code that ran -- **unless
    ``dirty`` is true**, in which case the working tree had uncommitted changes
    and the link shows the committed version instead. Filter on ``git.dirty``
    before trusting a permalink for reproducibility.
    """

    def run(*args):
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=repo_root(),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    # `git remote get-url` arrived in git 2.7; Sherlock ships 1.8.3.1, where it
    # is an unknown subcommand and this returned None -- so commit_url and
    # script_url, the whole point of recording the remote, were silently absent
    # from every record. `git config --get` predates both and is what get-url
    # reads anyway.
    remote = run("config", "--get", "remote.origin.url")
    if remote is None:
        remote = run("remote", "get-url", "origin")
    repo_url = normalize_remote_url(remote)
    return {
        "commit": commit,
        "short_commit": commit[:8] if commit else None,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
        "remote_url": repo_url,
        "commit_url": f"{repo_url}/commit/{commit}" if repo_url and commit else None,
    }


def script_url(script, commit: str | None, repo_url: str | None) -> str | None:
    """GitHub permalink to ``script`` at ``commit``, as a repo-relative blob URL."""
    if not (script and commit and repo_url):
        return None
    try:
        rel = Path(script).resolve().relative_to(repo_root())
    except (ValueError, OSError):
        return None
    return f"{repo_url}/blob/{commit}/{rel.as_posix()}"


def username() -> str | None:
    """The account that ran the step.

    ``$USER`` alone is unreliable: it is unset in a bare container, in some
    ``srun`` contexts, and under a cron-style launcher. ``getpass.getuser()``
    tries LOGNAME/USER/LNAME/USERNAME and then falls back to the password
    database, which is what actually works on a compute node.
    """
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        # No passwd entry for the uid -- happens in some container images.
        return os.environ.get("USER") or os.environ.get("LOGNAME")


def user_info() -> dict:
    """Who ran this, at both the account and the person level.

    ``user``/``uid`` identify the cluster account; ``git_user_name`` and
    ``git_user_email`` come from git config and identify the person, which is
    what you actually want when several people share a lab account or when an
    account name is opaque. Any of them may be None.
    """

    def git_config(key):
        try:
            out = subprocess.run(
                ["git", "config", "--get", key],
                cwd=repo_root(),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return out.stdout.strip() or None if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    try:
        uid = os.getuid()
    except AttributeError:  # non-POSIX
        uid = None

    return {
        "user": username(),
        "uid": uid,
        "git_user_name": git_config("user.name"),
        "git_user_email": git_config("user.email"),
    }


def slurm_info() -> dict:
    """Whatever SLURM put in the environment; all None outside a job."""

    def env(name):
        return os.environ.get(name) or None

    return {
        "job_id": env("SLURM_JOB_ID"),
        "array_job_id": env("SLURM_ARRAY_JOB_ID"),
        "array_task_id": env("SLURM_ARRAY_TASK_ID"),
        "job_name": env("SLURM_JOB_NAME"),
        "partition": env("SLURM_JOB_PARTITION"),
        "cpus": env("SLURM_CPUS_PER_TASK"),
        # Who submitted the job -- not always the process owner.
        "job_user": env("SLURM_JOB_USER"),
        "mem": env("SLURM_MEM_PER_NODE"),
        "nodelist": env("SLURM_JOB_NODELIST"),
    }


def tool_versions(extra: dict[str, str] | None = None) -> list[dict]:
    """Python plus whichever tracked distributions are installed."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as dist_version

    tools = [
        {"software_name": "python", "software_version": platform.python_version()},
    ]
    for name in TRACKED_DISTRIBUTIONS:
        try:
            tools.append({"software_name": name, "software_version": dist_version(name)})
        except PackageNotFoundError:
            continue
        revision = source_revision(name)
        if revision:
            tools.append({"software_name": f"{name}_commit", "software_version": revision})
    for name, ver in (extra or {}).items():
        tools.append({"software_name": name, "software_version": str(ver)})
    return tools


def source_revision(name: str) -> str | None:
    """The git commit a distribution was installed from, or None.

    A version string cannot tell unreleased builds apart -- chrombpnet 2.x is
    ``2.0.0.dev0`` at every commit, and MotifCompendium is installed from git.
    The installer's PEP 610 record (``direct_url.json``) can.
    """
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        raw = distribution(name).read_text("direct_url.json")
    except PackageNotFoundError:
        return None
    if not raw:
        return None
    try:
        return revision_from_direct_url(json.loads(raw))
    except (ValueError, AttributeError):
        return None


def revision_from_direct_url(info: dict) -> str | None:
    """Commit for a git install (``vcs_info``) or a local checkout (``dir_info``).

    A checkout is asked with git, and gets ``-dirty`` appended when tracked
    files differ from that commit (``-unverified`` when git cannot say), so a
    record never claims a commit the code did not match. ``cwd=`` rather than ``git -C``: Sherlock's git is 1.8.3.1,
    which predates -C.
    """
    commit = (info.get("vcs_info") or {}).get("commit_id")
    if commit:
        return commit
    url = info.get("url") or ""
    if "dir_info" not in info or not url.startswith("file://"):
        return None
    path = unquote(url[len("file://") :])

    def run(*args):
        try:
            out = subprocess.run(
                ["git", *args], cwd=path, capture_output=True, text=True, timeout=10, check=False
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    if not commit:
        return None
    status = run("status", "--porcelain", "--untracked-files=no")
    if status is None:  # git status failed or timed out: cannot vouch for a clean tree
        return commit + "-unverified"
    return commit + ("-dirty" if status else "")


class StepMetadata:
    """Accumulates one step's provenance record. Use via :func:`record`."""

    def __init__(self, step: str, dataset: str | None = None, out_dir=None, script=None):
        self.step = step
        self.dataset = dataset
        self.out_dir = Path(out_dir) if out_dir else None
        # The file implementing this step. Defaults to the running script; a
        # bash step passes its own path so the permalink points at the wrapper.
        self.script = (
            Path(script) if script else (Path(sys.argv[0]) if sys.argv and sys.argv[0] else None)
        )
        self.run_id = uuid.uuid4().hex
        self.started_monotonic = time.monotonic()
        self.started_at = _utc(time.time())
        self._inputs: list[tuple[str, Path]] = []
        self._outputs: list[tuple[str, Path]] = []
        self.params: list[dict] = []
        self.extra_tools: dict[str, str] = {}
        self.status = "ok"
        self.error: str | None = None
        self.exit_status = 0
        # Peak RSS is only meaningful when the process filling in this record
        # is the one that did the work. `record()` sets it on exit; the bash
        # path leaves it None unless the step measured it and passed it in,
        # because emit_metadata.py runs as a SIBLING of the real work and
        # would otherwise report its own ~26 MB as the step's footprint. Same
        # reasoning as `started_at` in emit_metadata.py.
        self.peak_rss_gb: float | None = None

    # -- declaration -----------------------------------------------------
    def add_input(self, role: str, path) -> None:
        self._inputs.append((role, Path(path)))

    def add_output(self, role: str, path) -> None:
        """Declare an output. It is hashed on exit, not now."""
        self._outputs.append((role, Path(path)))

    def add_param(self, key: str, value) -> None:
        """A setting that CONTROLLED the run (assay, plus_shift, seed)."""
        self._add_scalar(key, value, "param")

    def add_metric(self, key: str, value) -> None:
        """A quantity the run MEASURED (reads_kept, n_peaks). Not an input."""
        self._add_scalar(key, value, "metric")

    def _add_scalar(self, key: str, value, parameter_type: str) -> None:
        self.params.append(
            {
                "parameter_name": str(key),
                "parameter_value": "" if value is None else str(value),
                "parameter_type": parameter_type,
                "file": None,
                "filepath": None,
                "file_format": None,
                "file_exists": None,
                "file_size": None,
                "file_mtime": None,
                "md5sum": None,
                "md5_skipped": None,
            }
        )

    def add_params(self, mapping) -> None:
        for k, v in dict(mapping).items():
            self.add_param(k, v)

    def add_tool(self, name: str, version) -> None:
        self.extra_tools[str(name)] = str(version)

    # -- emission --------------------------------------------------------
    def to_dict(self) -> dict:
        git = git_info()
        who = user_info()
        try:
            rel_script = (
                str(Path(self.script).resolve().relative_to(repo_root())) if self.script else None
            )
        except (ValueError, OSError):
            rel_script = str(self.script) if self.script else None
        return {
            "schema_version": SCHEMA_VERSION,
            "uuid": self.run_id,
            "step": self.step,
            "dataset": self.dataset,
            "run_status": self.status,
            "exit_status": self.exit_status,
            "error": self.error,
            "date_created": self.started_at,
            "date_completed": _utc(time.time()),
            "duration_s": round(time.monotonic() - self.started_monotonic, 3),
            # Peak RSS, so SLURM --mem can be sized from measurement rather
            # than from a guess that gets copied forward. SELF plus CHILDREN,
            # because the heavy work is often a subprocess (chrombpnet,
            # modisco, bedtools). ru_maxrss is KiB on Linux, bytes on macOS.
            "peak_rss_gb": self.peak_rss_gb,
            "host": socket.gethostname(),
            # Flat scalars: `GROUP BY "user"` is the common query, so these do
            # not get buried in a struct.
            "user": who["user"],
            "uid": who["uid"],
            "git_user_name": who["git_user_name"],
            "git_user_email": who["git_user_email"],
            "cwd": os.getcwd(),
            "command": list(sys.argv),
            "checksums_enabled": checksums_enabled(),
            "script": rel_script,
            "script_url": script_url(self.script, git.get("commit"), git.get("remote_url")),
            "git": git,
            "slurm": slurm_info(),
            # ONE long-format list. parameter_type tells the four kinds apart:
            #   input   a file the run consumed
            #   output  a file the run produced
            #   param   a setting that controlled it
            #   metric  a quantity it measured
            # UNNEST gives a flat table where every column is meaningful on its
            # own -- which is why nothing here is called `key`, `value`, `name`
            # or `path`.
            "parameters": (
                [file_record(r, p, parameter_type="input") for r, p in self._inputs]
                + [file_record(r, p, parameter_type="output") for r, p in self._outputs]
                + self.params
            ),
            "software_versions": tool_versions(self.extra_tools),
        }

    def path(self) -> Path:
        # Flat: one directory, one file per invocation. The step is in the
        # FILENAME, not a parent directory, because the directory tree was
        # carrying two namespaces at once -- numbered job records
        # ("00.0.prepare_signal") next to unnumbered tool records
        # ("prepare_bigwig") -- which reads as an inconsistency when you list
        # the directory, even though it is the deliberate job/tool split that
        # queries.sql turns into its `jobs` and `tools` views. `step` is a
        # column in every record, so DuckDB does the filtering either way, and
        # a flat glob is simpler. The cost is that you can no longer count a
        # step's runs with `ls metadata/<step>/`; use the `runs` view.
        stamp = self.started_at.replace(":", "").replace("-", "")
        return Path(self.out_dir) / f"{stamp}_{self.step}_{self.run_id[:8]}.json"

    def write(self, path=None) -> Path | None:
        """Write the record. Never raises -- provenance must not fail a step."""
        if self.out_dir is None and path is None:
            logger.debug("no metadata dir configured, not writing a record")
            return None
        target = Path(path) if path else self.path()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        except OSError as exc:
            logger.warning("could not write run metadata to %s: %s", target, exc)
            return None
        logger.info("run metadata -> %s", target)
        return target


@contextmanager
def record(step: str, dataset: str | None = None, out_dir=None, script=None):
    """Context manager that writes a step's metadata on the way out.

    Writes on success *and* on failure, then re-raises. Outputs are hashed at
    exit, so declare them as soon as their paths are known.
    """
    md = StepMetadata(step, dataset=dataset, out_dir=out_dir, script=script)
    try:
        yield md
    except BaseException as exc:  # noqa: BLE001  # recorded, then re-raised
        md.status = "failed"
        md.error = f"{type(exc).__name__}: {exc}"
        md.exit_status = (
            exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 1
        )
        if md.exit_status == 0:
            md.status = "ok"
            md.error = None
        md.peak_rss_gb = _peak_rss_gb()
        md.write()
        raise
    else:
        md.peak_rss_gb = _peak_rss_gb()
        md.write()


def default_dir(results_path=None) -> Path | None:
    """``<results_path>/metadata``, or ``$METADATA_DIR`` if set."""
    env = os.environ.get("METADATA_DIR")
    if env:
        return Path(env)
    if results_path:
        return Path(results_path) / "metadata"
    return None


def report_peak_rss() -> None:
    """Write this process's peak RSS (GiB) to $METADATA_RSS_FILE, if the step asked.

    The process that allocates the big arrays is the one whose peak is the
    step's real footprint. A bash step's metadata trap runs in a SIBLING
    (emit_metadata.py), which cannot see it, so the number is handed over
    through a file the step then reads into `metadata_metrics`. Children the
    process waited for count too (RUSAGE_CHILDREN) -- TF-MoDISco runs as one.
    See docs/resource-measurements.md.
    """
    target = os.environ.get("METADATA_RSS_FILE")
    if not target:
        return
    try:
        import resource

        scale = 1 if sys.platform == "darwin" else 1024  # macOS reports bytes
        peak = max(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        )
        Path(target).write_text(f"{peak * scale / 2**30:.3f}\n")
    except Exception:  # noqa: BLE001  # provenance must not fail a step
        logging.getLogger(__name__).debug("could not record peak RSS", exc_info=True)
