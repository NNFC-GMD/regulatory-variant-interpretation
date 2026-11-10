"""workflows/pipeline/project.py: project.yaml's validation, the layout it derives, the stage checks and the files it
writes for DeepISA; and fetch_igvf.py's offline parts (no network)."""

from __future__ import annotations

import gzip
import hashlib
import json
import shlex
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

PIPELINE = Path(__file__).resolve().parents[1] / "workflows" / "pipeline"
sys.path.insert(0, str(PIPELINE))

import fetch_igvf as fi  # noqa: E402
import project as pj  # noqa: E402


def _yaml(tmp_path: Path, **over) -> Path:
    doc = {
        "project": "toy",
        "root": str(tmp_path / "root"),
        "prefix": "tx_",
        "compendium": "tx2",
        "samples": [
            {"stem": "a", "fragments": "IGVFFI0000AAAA", "index": "IGVFFI0000AAAB"},
            {"stem": "b", "fragments": str(tmp_path / "b.fragments.tsv.gz")},
        ],  # fmt: skip
        "cooperativity": {"comparisons": [["a_b", "a", "b"]]},
        "reference_root": str(tmp_path / "refs"),
    }
    doc.update(over)
    doc = {k: v for k, v in doc.items() if v is not None}  # over=None drops a key
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


GEFION = {"cpus": 216, "gpus": [str(g) for g in range(8)], "mem_gb": 2000}
"""A whole Gefion node, the machine SLOTS was sized on."""


def test_load_fills_defaults_and_derives_names(tmp_path):
    p = pj.load(_yaml(tmp_path))
    assert p["datasets"] == ["tx_a", "tx_b"] and p["folds"] == ["0", "1", "2", "3", "4"]
    assert p["stages"] == list(pj.BOX_STAGES) and p["slots"] == {} and p["slurm"] == {}
    # every reference from reference_root, through the vendored layout; the folds are the vendored ones
    assert p["genome"] == str(tmp_path / "refs" / "hg38" / "Sequence" / "IGVFFI0653VCGH.fasta")
    assert Path(p["folds_dir"]) == pj.VENDOR / "folds" and (pj.VENDOR / "folds" / "fold_4.json").is_file()
    assert p["samples"][0]["accession"] == "IGVFFI0000AAAA" and p["samples"][1]["accession"] is None
    assert p["cooperativity"]["groups"] == {"a": ["tx_a"], "b": ["tx_b"]}  # one group per sample by default
    q = pj.paths(p, "tx_a")
    root = tmp_path / "root"
    assert q["narrowpeak"] == root / "peaks" / "tx_a" / "tx_a.narrowPeak.gz"
    assert q["finemo"] == root / "cherimoya" / "tx_a" / "downstream" / "finemo_tx2" / "tx_a_all"
    assert pj.paths(p)["motif_list"] == root / "compendia" / "inputs" / "cheri2.tsv"
    assert pj.fragments_file(p, p["samples"][0]) == root / "fragments" / "a.fragments.tsv.gz"


def test_stages_are_kept_in_pipeline_order(tmp_path):
    p = pj.load(_yaml(tmp_path, stages=["cooperativity", "prep"]))
    assert p["stages"] == ["prep", "cooperativity"]


@pytest.mark.parametrize(
    "over, message",
    [
        ({"root": "relative/path"}, "absolute"),
        ({"compendium": "bad-tag"}, "must match"),
        ({"stages": ["prep", "fetch"]}, "unknown stages"),
        ({"samples": [{"stem": "a", "fragments": "not_a_path"}]}, "neither an IGVF file accession"),
        (
            {"samples": [{"stem": "a", "fragments": "IGVFFI0000AAAA"}, {"stem": "a", "fragments": "IGVFFI0000AAAB"}]},
            "not unique",
        ),
        ({"cooperativity": {"comparisons": [["x", "a", "zz"]]}}, "no group zz"),
        ({"cooperativity": {"groups": {"g": ["nope"]}}}, "unknown stems"),
        ({"slots": {"gpus": 3}}, "unknown slots"),
        ({"slots": {"train_cores": 0}}, "positive integer"),
        ({"surprise": 1}, "unknown keys"),
        ({"igvf": "/x/igvf_tf_chrombpnet"}, "no longer project.yaml keys"),
        ({"genome": "/x/genome.fa", "folds_dir": "/x/folds"}, "reference_root instead"),
        ({"chrombpnet": "/x/chrombpnet", "chrombpnet_env": "cuda13"}, "pixi env `chrombpnet`"),
        ({"reference_root": "relative/refs"}, "reference_root must be an absolute"),
        ({"slurm": {"queue": "defq"}}, "slurm takes only"),
    ],
)
def test_invalid_projects_are_refused(tmp_path, over, message):
    with pytest.raises(pj.ProjectError, match=message):
        pj.load(_yaml(tmp_path, **over))


def test_env_lines_are_shell_assignments(tmp_path):
    p = pj.load(_yaml(tmp_path))
    env = dict(line.split("=", 1) for line in pj.env_lines(p, GEFION))
    assert shlex.split(env["PROJECT_DATASETS"]) == ["tx_a tx_b"]
    assert shlex.split(env["PROJECT_COMPENDIUM"]) == [str(tmp_path / "root" / "compendia" / "tx2")]
    assert env["SLOT_FINEMO_SLOTS"] == "5" and env["PROJECT_N_DATASETS"] == "2"
    assert env["PROJECT_GPUS"] == "0,1,2,3,4,5,6,7" and env["PROJECT_CPUS"] == "216"
    assert shlex.split(env["PROJECT_REFERENCE_ROOT"]) == [str(tmp_path / "refs")]
    refs = shlex.split(env["PROJECT_REFERENCE_FILES"])
    assert len(refs) == len(pj.REFERENCE_FILES) + 1 and refs[-1] == p["genome"] + ".fai"
    assert shlex.split(env["PROJECT_VENDOR"]) == [str(pj.VENDOR)]


def test_running_needs_reference_root_but_reading_does_not(tmp_path, capsys):
    path = _yaml(tmp_path, reference_root=None)
    p = pj.load(path)
    assert p["genome"] is None and pj.check(p, "fetch")  # check, status, prune and share need no references
    with pytest.raises(pj.ProjectError, match="reference_root is required"):
        pj.env_lines(p, GEFION)
    with pytest.raises(pj.ProjectError, match="reference_root is required"):
        pj.write_coop_inputs(p)
    assert pj.main([str(path), "env"]) == 2 and "reference_root" in capsys.readouterr().err
    assert pj.main([str(path), "status"]) == 0


def test_slots_scale_to_the_machine():
    # a whole Gefion node: SLOTS as they are, and every dataset's prep chain at once
    assert pj.resolve_slots({}, GEFION, 5) == {**pj.SLOTS, "prep_max": 5}
    # a 1-GPU workstation of 16 CPUs and 64 GB
    small = pj.resolve_slots({}, {"cpus": 16, "gpus": ["0"], "mem_gb": 64}, 5)
    assert small == {"train_per_gpu": 4, "train_cores": 4, "attribute_per_gpu": 4, "attribute_cores": 4,
                     "modisco_max": 1, "modisco_threads": 16, "finemo_slots": 1, "finemo_batch": 200,
                     "coop_cores": 16, "prep_max": 1}  # fmt: skip
    # half a node: the per-slot cores stay, the at-once counts shrink
    half = pj.resolve_slots({}, {"cpus": 96, "gpus": ["0", "1", "2", "3"], "mem_gb": 1000}, 5)
    assert half["train_cores"] == 6 and half["modisco_max"] == 4 and half["finemo_slots"] == 4 and half["prep_max"] == 5
    # whatever the machine, the GPU stages fit its CPUs
    for m in ({"cpus": 8, "gpus": ["0"], "mem_gb": 32}, {"cpus": 64, "gpus": ["0", "1"], "mem_gb": 0}, GEFION):
        s = pj.resolve_slots({}, m, 3)
        for k in ("train", "attribute"):
            assert len(m["gpus"]) * s[f"{k}_per_gpu"] * s[f"{k}_cores"] <= m["cpus"]
    # a value project.yaml gives is kept, and the rest follow from it
    given = pj.resolve_slots({"train_per_gpu": 2, "prep_max": 0, "modisco_max": 2}, GEFION, 7)
    assert given["train_per_gpu"] == 2 and given["train_cores"] == 6 and given["prep_max"] == 7
    assert given["modisco_max"] == 2 and given["modisco_threads"] == 24


def test_machine_reads_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    m = pj.machine()
    assert m["gpus"] == ["2", "3"] and m["cpus"] >= 1 and m["mem_gb"] >= 0


def test_slurm_block_becomes_sbatch_options(tmp_path):
    p = pj.load(_yaml(tmp_path, slurm={"partition": "gpu", "account": "lab", "gpus": 4, "cpus": 64,
                                       "time": "1-00:00:00", "extra": "--qos=high --constraint=h100"}))  # fmt: skip
    assert pj.sbatch_args(p) == ["--partition=gpu", "--account=lab", "--gres=gpu:4", "--cpus-per-task=64",
                                 "--time=1-00:00:00", "--qos=high", "--constraint=h100"]  # fmt: skip
    assert pj.sbatch_args(pj.load(_yaml(tmp_path))) == []


def _touch(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_checks_follow_the_outputs(tmp_path):
    p = pj.load(_yaml(tmp_path, folds=["0"]))
    assert pj.check(p, "fetch") == [str(pj.fragments_file(p, s)) for s in p["samples"]]
    for s in p["samples"]:
        _touch(pj.fragments_file(p, s))
    assert pj.check(p, "fetch") == []
    missing = pj.check(p, "prep")
    assert len(missing) == 2 * 6  # config, narrowPeak, bigwig, peaks, signal QC, one fold's negatives
    for ds in p["datasets"]:
        q = pj.paths(p, ds)
        for key in ("config", "narrowpeak", "bigwig", "peaks", "signal_qc"):
            _touch(q[key])
        _touch(pj.negatives(p, ds, "0"))
    assert pj.check(p, "prep") == []
    # an empty file is not an output
    pj.paths(p, "tx_a")["bigwig"].write_text("")
    assert pj.check(p, "prep") == [str(pj.paths(p, "tx_a")["bigwig"])]

    for ds in p["datasets"]:
        q = pj.paths(p, ds)
        for m in ("06.0", "v2", "07.0"):
            _touch(q["downstream"] / ".done" / m)
    assert pj.check(p, "downstream") == [str(pj.paths(p, ds)["downstream"] / ".done" / "08.0") for ds in p["datasets"]]

    assert str(pj.paths(p)["compendium"] / "compendium.json") == pj.check(p, "motifs")[0]
    with pytest.raises(pj.ProjectError):
        pj.check(p, "nonsense")


def test_coop_inputs(tmp_path):
    p = pj.load(_yaml(tmp_path, folds=["0", "1"]))
    written = pj.write_coop_inputs(p)
    coop = tmp_path / "root" / "cherimoya" / "cooperativity"
    args = (coop / "args" / "tx_b.args").read_text().splitlines()
    assert args[:2] == ["--dataset", "tx_b"] and args[-3:] == ["--folds", "0", "1"]
    assert args[args.index("--models") + 1].endswith("tx_b/fold_{fold}/cherimoya.fold_{fold}.torch")
    assert args[args.index("--finemo-dir") + 1].endswith("tx_b/downstream/finemo_tx2/tx_b_all")
    assert (coop / "groups.tsv").read_text() == "group\tdatasets\na\ttx_a\nb\ttx_b\n"
    assert (coop / "comparisons.tsv").read_text() == "name\tgroup_a\tgroup_b\na_b\ta\tb\n"
    assert coop / "groups.tsv" in written


def test_the_example_project_loads():
    p = pj.load(PIPELINE / "project.example.yaml")
    assert p["datasets"] == ["ex_t0", "ex_t1", "ex_t2"] and len(p["cooperativity"]["comparisons"]) == 3


def test_main_exit_codes(tmp_path, capsys):
    path = _yaml(tmp_path)
    assert pj.main([str(path), "check", "fetch"]) == 1
    assert "missing:" in capsys.readouterr().out
    bad = tmp_path / "bad.yaml"
    bad.write_text("project: x\n")
    assert pj.main([str(bad), "show"]) == 2
    # driver: false (a project set up by hand, kept for its results) loads without reference_root
    by_hand = _yaml(tmp_path, driver=False, reference_root=None)
    assert pj.main([str(by_hand), "status"]) == 0
    assert pj.main([str(by_hand), "env"]) == pj.DRIVER_REFUSED


# fetch_igvf.py, offline


def _bgzf(path: Path, text: str = "chr1\t1\t2\tAC-1\t1\n") -> Path:
    # a BGZF block: gzip with FEXTRA and the 'BC' subfield (what bgzip writes); Python's gzip cannot add it, so the
    # header is written by hand around a raw deflate stream
    import zlib

    data = text.encode()
    comp = zlib.compressobj(9, zlib.DEFLATED, -15)
    body = comp.compress(data) + comp.flush()
    bsize = 12 + 6 + len(body) + 8 - 1
    header = b"\x1f\x8b\x08\x04" + b"\x00" * 4 + b"\x00\xff" + b"\x06\x00" + b"BC\x02\x00" + bsize.to_bytes(2, "little")
    trailer = zlib.crc32(data).to_bytes(4, "little") + len(data).to_bytes(4, "little")
    path.write_bytes(header + body + trailer)
    return path


def test_bgzf_detection(tmp_path):
    assert fi.is_bgzf(_bgzf(tmp_path / "x.fragments.tsv.gz"))
    plain = tmp_path / "plain.gz"
    with gzip.open(plain, "wt") as fh:
        fh.write("chr1\t1\t2\n")
    assert not fi.is_bgzf(plain)
    with gzip.open(_bgzf(tmp_path / "y.gz", "chr2\t5\t9\tB\t1\n"), "rt") as fh:  # and it is valid gzip
        assert fh.read().startswith("chr2")


def test_fetch_one_keeps_a_file_in_place_and_refuses_a_wrong_md5(tmp_path, monkeypatch):
    dest = tmp_path / "d.bin"
    dest.write_bytes(b"payload")
    md5 = hashlib.md5(b"payload").hexdigest()
    meta = {"accession": "IGVFFI0000AAAA", "md5sum": md5, "href": "/x/@@download/x.bed.gz", "file_size": 7,
            "summary": "toy"}  # fmt: skip
    monkeypatch.setattr(fi, "metadata", lambda acc: meta)
    monkeypatch.setattr(fi, "download", lambda *a: pytest.fail("a file in place with the right md5 is not fetched"))
    assert fi.fetch_one("IGVFFI0000AAAA", dest, None, dry=False)["state"] == "kept"
    with pytest.raises(RuntimeError, match="differs from the portal's"):
        fi.fetch_one("IGVFFI0000AAAA", dest, "0" * 32, dry=False)


def test_main_links_a_local_bgzf_file_and_writes_the_manifest(tmp_path, monkeypatch):
    path = _yaml(tmp_path, samples=[{"stem": "b", "fragments": str(tmp_path / "b.fragments.tsv.gz")}], cooperativity={})
    _bgzf(tmp_path / "b.fragments.tsv.gz")
    assert fi.main([str(path)]) == 0
    link = tmp_path / "root" / "fragments" / "b.fragments.tsv.gz"
    assert link.is_symlink() and link.resolve() == (tmp_path / "b.fragments.tsv.gz").resolve()
    manifest = (tmp_path / "root" / "portal" / "manifest.tsv").read_text().splitlines()
    assert manifest[0].split("\t")[:3] == ["accession", "stem", "role"] and manifest[1].split("\t")[2] == "fragments"
    p = pj.load(path)
    assert pj.check(p, "fetch") == []
    json.dumps(pj.load(path))  # the project serialises (project.py show)
