"""vendor/igvf_tf_chrombpnet: the copy of igvf_tf_chrombpnet's steps is exactly what MANIFEST.tsv records, it is
complete (every file the copied steps source, call or import is in it), and what this repository runs it with
(environments, chrombpnet's commit, the references layout) agrees with it. Standard library only."""

from __future__ import annotations

import ast
import hashlib
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "vendor" / "igvf_tf_chrombpnet"
NOT_COPIED = {"VENDORED.md", "MANIFEST.tsv"}
"""The copy's own files: written here, not copied."""
DRIVER_ENVS = {"peaks", "preprocess", "modisco", "motif-compendium", "finemo"}
"""The vendored environments run_project.sh and setup.sh use."""
UNREACHED = (
    re.compile(r"^config(/|$)"),  # dataset configs by name; the driver passes each config's path
    re.compile(r"^results(/|$)"),  # igvf's own output tree; every step gets its project's output_dir
    re.compile(r"^\.pixi/"),  # an installed environment, not a file of the copy
    re.compile(r"^workflows/SLURM/status\.sh$"),  # named in an error message
    re.compile(r"^workflows/SLURM/0[345]\.\d\.[a-z_]+\.sh$"),  # bias and ChromBPNet steps; CHAIN_STAGES=prep skips them
)
"""Paths the copied scripts name that the driver never reaches, so they are not copied."""


def manifest() -> tuple[dict[str, str], dict[str, str]]:
    header, rows = {}, {}
    for line in (VENDOR / "MANIFEST.tsv").read_text().splitlines():
        if line.startswith("# "):
            key, _, value = line[2:].partition("\t")
            header[key] = value
        elif line and line != "path\tsha256":
            path, digest = line.split("\t")
            rows[path] = digest
    return header, rows


def test_manifest_names_a_commit_and_every_file_matches():
    header, rows = manifest()
    assert re.fullmatch(r"[0-9a-f]{40}", header["commit"]) and header["source"].endswith("IGVF/igvf_tf_chrombpnet")
    bad = [p for p, d in rows.items() if hashlib.sha256((VENDOR / p).read_bytes()).hexdigest() != d]
    assert bad == [], f"edited after vendoring (refresh with workflows/pipeline/vendor_igvf.sh): {bad}"
    files = {
        str(p.relative_to(VENDOR))
        for p in VENDOR.rglob("*")
        if p.is_file() and ".pixi" not in p.relative_to(VENDOR).parts and "__pycache__" not in p.parts
    }
    assert files - NOT_COPIED == set(rows), "files under vendor/ that MANIFEST.tsv does not list, or the reverse"


def _unreached(path: str) -> bool:
    return any(r.search(path) for r in UNREACHED)


def test_every_path_the_copied_scripts_name_is_copied():
    """${REPO_ROOT}/..., ${src_dir}/... and ${steps_dir}/... in the shell files, and the steps run_chain.sh and
    motif_chain.sh run by name."""
    _, rows = manifest()
    prefixes = {"REPO_ROOT": "", "src_dir": "src/", "steps_dir": "workflows/SLURM/"}
    missing = []
    for rel in (p for p in rows if p.endswith(".sh")):
        text = (VENDOR / rel).read_text()
        for var, path in re.findall(r"\$\{(REPO_ROOT|src_dir|steps_dir)\}/([A-Za-z0-9_./-]+)", text):
            target = (prefixes[var] + path).rstrip("./")
            if "${" in target or _unreached(target):
                continue
            if not ((VENDOR / target).exists() or any(r.startswith(target + "/") for r in rows)):
                missing.append(f"{rel}: {var}/{path}")
        for step in re.findall(r"\b(\d\d\.\d\.[a-z_]+)(?:\.sh)?\b", text):
            target = f"workflows/SLURM/{step}.sh"
            if rel.startswith("workflows/dcai/") and not _unreached(target) and target not in rows:
                missing.append(f"{rel}: step {step}")
    assert missing == []


def test_every_import_of_the_copied_python_resolves():
    _, rows = manifest()
    utils = VENDOR / "lib" / "python" / "utils"
    missing = []
    for rel in (p for p in rows if p.endswith(".py")):
        tree = ast.parse((VENDOR / rel).read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "utils":
                parts = node.module.split(".")[1:]
                names = [parts[0]] if parts else [a.name for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name.split(".")[1] for a in node.names if a.name.startswith("utils.")]
            missing += [f"{rel}: utils.{n}" for n in names if not (utils / f"{n}.py").is_file()]
    assert missing == []


def test_the_environments_the_driver_uses_are_the_copy_s_and_ours():
    igvf = tomllib.loads((VENDOR / "pixi.toml").read_text())
    assert DRIVER_ENVS <= set(igvf["environments"])
    ours = tomllib.loads((REPO / "pixi.toml").read_text())
    assert {"default", "cherimoya", "chrombpnet"} <= set(ours["environments"]) | {"default"}
    # chrombpnet at the commit the copied steps name (CHROMBPNET_REV in lib/bash/common.sh)
    common = (VENDOR / "lib" / "bash" / "common.sh").read_text()
    rev = re.search(r'CHROMBPNET_REV="\$\{CHROMBPNET_REV:-([0-9a-f]{40})\}"', common)
    assert rev and ours["feature"]["chrombpnet"]["pypi-dependencies"]["chrombpnet"]["rev"] == rev.group(1)
    for script in ("run_project.sh", "setup.sh"):
        text = (REPO / "workflows" / "pipeline" / script).read_text()
        assert all(e in text for e in DRIVER_ENVS), script


def test_env_sh_finds_the_copy_and_keeps_what_the_driver_sets(tmp_path):
    out = subprocess.run(
        ["bash", "-c", 'source "$1" && printf "%s\\n" "${REPO_ROOT}" "${REFERENCE_ROOT}" "${DATASET_ROOT}"', "_",
         str(VENDOR / "workflows" / "dcai" / "env.sh")],
        env={"PATH": os.environ["PATH"], "HOME": str(tmp_path), "REFERENCE_ROOT": "/refs", "DATASET_ROOT": "/proj"},
        capture_output=True, text=True, check=True,
    ).stdout.split()  # fmt: skip
    assert out == [str(VENDOR), "/refs", "/proj"]


def test_project_takes_the_genome_from_the_copy_s_layout(tmp_path):
    sys.path.insert(0, str(REPO / "workflows" / "pipeline"))
    import project

    refs = project.reference_files(tmp_path)
    assert refs["genome_fa"] == str(tmp_path / "hg38" / "Sequence" / "IGVFFI0653VCGH.fasta")
    assert all(Path(v).is_relative_to(tmp_path) for v in refs.values())
    assert project.VENDOR == VENDOR and (VENDOR / "folds" / "fold_0.json").is_file()
