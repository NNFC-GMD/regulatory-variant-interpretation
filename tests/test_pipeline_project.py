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
    }
    doc.update(over)
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def test_load_fills_defaults_and_derives_names(tmp_path):
    p = pj.load(_yaml(tmp_path))
    assert p["datasets"] == ["tx_a", "tx_b"] and p["folds"] == ["0", "1", "2", "3", "4"]
    assert p["stages"] == list(pj.BOX_STAGES) and p["slots"]["train_per_gpu"] == 4
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
        ({"surprise": 1}, "unknown keys"),
    ],
)
def test_invalid_projects_are_refused(tmp_path, over, message):
    with pytest.raises(pj.ProjectError, match=message):
        pj.load(_yaml(tmp_path, **over))


def test_env_lines_are_shell_assignments(tmp_path):
    p = pj.load(_yaml(tmp_path))
    env = dict(line.split("=", 1) for line in pj.env_lines(p))
    assert shlex.split(env["PROJECT_DATASETS"]) == ["tx_a tx_b"]
    assert shlex.split(env["PROJECT_COMPENDIUM"]) == [str(tmp_path / "root" / "compendia" / "tx2")]
    assert env["SLOT_FINEMO_SLOTS"] == "5" and env["PROJECT_N_DATASETS"] == "2"


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
