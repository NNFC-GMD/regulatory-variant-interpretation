"""workflows/pipeline/prune.py and share.py: which files a finished project loses and which it keeps, the manifest
and its apply step, the driver: false guard, and the share tar. Synthetic trees only (standard library and PyYAML)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

PIPELINE = Path(__file__).resolve().parents[1] / "workflows" / "pipeline"
import sys  # noqa: E402

sys.path.insert(0, str(PIPELINE))

import project as pj  # noqa: E402
import prune  # noqa: E402
import share  # noqa: E402


def _yaml(tmp_path: Path, **over) -> Path:
    doc = {
        "project": "toy",
        "root": str(tmp_path / "root"),
        "prefix": "tx_",
        "compendium": "tx2",
        "folds": ["0"],
        "samples": [
            {"stem": "a", "fragments": str(tmp_path / "a.fragments.tsv.gz")},
            {"stem": "b", "fragments": str(tmp_path / "b.fragments.tsv.gz")},
        ],
    }
    doc.update(over)
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def _put(root: Path, rel: str, text: str = "x\n") -> Path:
    f = root / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)
    return f


def _project_tree(p: dict) -> dict[str, list[str]]:
    """A finished project's files for every dataset: what prune must keep and what it must delete."""
    root = Path(p["root"])
    keep, delete = [], []
    for s in p["samples"]:
        keep.append(pj.fragments_file(p, s).relative_to(root).as_posix())
    keep += ["project.yaml", "NOTES.md", "STATUS.md", "compendia/tx2/compendium.json", "box/1/stages.tsv",
             "chrombpnet/shifts.json"]  # fmt: skip
    for ds in p["datasets"]:
        fold = f"cherimoya/{ds}/fold_0"
        keep += [f"{fold}/cherimoya.fold_0.torch", f"{fold}/cherimoya.fold_0.test.performance.tsv",
                 f"{fold}/attribute.json", f"{fold}/attr.attr.npz", f"{fold}/attr.ohe.npz", f"{fold}/attr.idx.npy",
                 f"{fold}/attr.attr.stock.json"]  # fmt: skip
        dn = f"cherimoya/{ds}/downstream"
        interp = f"{dn}/full_models/{ds}_all_fold_0/interpretation"
        keep += [f"{interp}/interpretation.interpreted_regions.bed", f"{interp}/interpretation.provenance.json"]
        delete += [f"{interp}/interpretation.counts_scores.h5"]
        keep += [f"{dn}/.done/{m}" for m in ("06.0", "v2", "07.0", "08.0")]
        keep += [f"{dn}/contrib_scores/{ds}/{ds}_average_shaps.counts.h5",
                 f"{dn}/contrib_scores/{ds}/modisco/modisco_counts_results.h5"]  # fmt: skip
        fm = f"{dn}/finemo_tx2/{ds}_all"
        keep += [f"{fm}/{f}" for f in ("hits.tsv", "hits_unique.tsv", "hits.bed.gz", "peaks_qc.tsv", "compendium.md5")]
        keep += [f"{fm}/finemo_report/motif_report.tsv", f"{fm}/finemo_report/fig/a.png"]
        delete += [f"{fm}/intermediate_inputs.npz", f"{fm}/hits.bed"]
        keep += [f"peaks/{ds}/{ds}.narrowPeak.gz", f"peaks/{ds}/call_peaks.json"]
        delete += [f"peaks/{ds}/reps/{ds}.rep1.narrowPeak", f"peaks/{ds}/reps/{ds}.repT.xls"]
        prep = f"chrombpnet/{ds}"
        keep += [f"chrombpnet/configs/{ds}/config.yaml", f"{prep}/preprocessing/signal/data_unstranded.bw",
                 f"{prep}/preprocessing/peaks/{ds}_all_peaks_no_blacklist.narrowPeak",
                 f"{prep}/preprocessing/{ds}/output_all_fold_0_negatives.bed",
                 f"{prep}/plots/signal_qc/{ds}_signal_qc.json"]  # fmt: skip
        delete += [f"{prep}/preprocessing/{ds}/output_all_fold_0_auxiliary/genomewide_gc.bed"]
        keep += [f"cherimoya/cooperativity/{ds}/run.json", f"cherimoya/cooperativity/{ds}/pairs.tsv.gz"]
    keep += ["cherimoya/cooperativity/combined/summary.json", "cherimoya/cooperativity/official/summary.json"]
    for rel in keep + delete:
        _put(root, rel)
    for ds in p["datasets"]:  # attribute.json names the fold's attribution files, as `cherimoya attribute` reads it
        fold = root / "cherimoya" / ds / "fold_0"
        (fold / "attribute.json").write_text(json.dumps({
            "model": str(fold / "cherimoya.fold_0.torch"), "attr_filename": str(fold / "attr.attr.npz"),
            "ohe_filename": str(fold / "attr.ohe.npz"), "idx_filename": str(fold / "attr.idx.npy")}))  # fmt: skip
    return {"keep": keep, "delete": delete}


def test_translate():
    rx = lambda pat: re.compile(prune.translate(pat, ["tx_a", "tx_ab"], ["tx_a"]))  # noqa: E731
    assert rx("**/*.md").fullmatch("a/b/c.md") and rx("**/*.md").fullmatch("c.md")
    assert not rx("*.md").fullmatch("a/c.md")
    assert rx("x/**").fullmatch("x/y/z") and not rx("x/**").fullmatch("xy/z")
    assert rx("cherimoya/<ds>/f").fullmatch("cherimoya/tx_ab/f") and not rx("cherimoya/<ds>/f").fullmatch(
        "cherimoya/tx_c/f"
    )
    assert rx("c/<shared>/f").fullmatch("c/tx_a/f") and not rx("c/<shared>/f").fullmatch("c/tx_ab/f")
    assert rx("d/*.{h5,bw}").fullmatch("d/s.bw") and not rx("d/*.{h5,bw}").fullmatch("d/s.bed")
    assert rx("e/{a/**,b.txt}").fullmatch("e/a/q/r") and rx("e/{a/**,b.txt}").fullmatch("e/b.txt")
    assert rx("f/?.txt").fullmatch("f/1.txt") and not rx("f/?.txt").fullmatch("f/12.txt")


def test_plan_deletes_the_intermediates_and_keeps_the_rest(tmp_path):
    p = pj.load(_yaml(tmp_path))
    tree = _project_tree(p)
    root = Path(p["root"])
    _put(root, "peaks/tx_a/reps/README.md")  # a note inside a deleted directory stays
    _put(root, "odd/thing.dat", "12345")  # no rule: kept and listed
    pl = prune.plan(p)
    deleted = sorted(rel for rel, _, _ in pl.delete)
    assert deleted == sorted(tree["delete"])
    assert [rel for rel, _ in pl.unclassified] == ["odd/thing.dat"]
    whole = {rel for rel, _ in pl.whole}
    assert "cherimoya/tx_a/fold_0" in whole  # kept whole without being walked
    assert "cherimoya/tx_a/downstream/finemo_tx2/tx_a_all/finemo_report" in whole  # motifs' check reads it
    files = prune.write_plan(p, pl, stamp="T")
    rows = files["manifest"].read_text().splitlines()
    assert rows[1] == f"# root\t{root}" and rows[2] == "path\tbytes\trule"
    assert len(rows) == 3 + len(tree["delete"])
    assert "unclassified" in files["summary"].read_text()


def test_what_a_check_reads_is_never_deleted(tmp_path):
    # no rule set and a project rule that deletes everything: only ALWAYS and what the checks read survive
    p = pj.load(_yaml(tmp_path, prune={"rulesets": [], "delete": ["**"]}))
    tree = _project_tree(p)
    pl = prune.plan(p)
    deleted = {rel for rel, _, _ in pl.delete}
    root = Path(p["root"])
    for stage in pj.STAGES:
        for path in pj.needed(p, stage):
            rel = Path(path).relative_to(root).as_posix()
            assert not any(d == rel or d.startswith(rel + "/") for d in deleted), (stage, rel)
    for ds in p["datasets"]:  # the attribution files attr_status reads through attribute.json
        for f in ("attr.attr.npz", "attr.ohe.npz", "attr.idx.npy", "attr.attr.stock.json", "cherimoya.fold_0.torch"):
            assert f"cherimoya/{ds}/fold_0/{f}" not in deleted
    assert "cherimoya/tx_a/downstream/contrib_scores/tx_a/tx_a_average_shaps.counts.h5" in deleted
    assert len(deleted) > len(tree["delete"])


def test_checks_still_pass_after_apply(tmp_path):
    p = pj.load(_yaml(tmp_path))
    tree = _project_tree(p)
    stages = ("fetch", "prep", "train", "downstream", "motifs", "cooperativity")
    before = {s: pj.check(p, s) for s in stages}
    assert all(v == [] for v in before.values()), before
    pl = prune.plan(p)
    c = prune.apply(p, prune.write_plan(p, pl, stamp="T")["manifest"], stamp="T")
    assert c["deleted"] == len(tree["delete"]) and c["refused"] == c["changed"] == 0
    root = Path(p["root"])
    assert all(not (root / rel).exists() for rel in tree["delete"])
    assert all((root / rel).exists() for rel in tree["keep"])
    assert not (root / "peaks" / "tx_a" / "reps").exists()  # left empty, so removed
    assert {s: pj.check(p, s) for s in stages} == before


def test_apply_refuses_what_it_should(tmp_path):
    p = pj.load(_yaml(tmp_path))
    _project_tree(p)
    root = Path(p["root"])
    outside = tmp_path / "outside.txt"
    outside.write_text("precious\n")
    (root / "peaks" / "tx_a" / "reps" / "link.narrowPeak").symlink_to(outside)  # a matched link: the link goes
    manifest = prune.write_plan(p, prune.plan(p), stamp="T")["manifest"]
    rows = manifest.read_text()
    (root / "peaks" / "tx_b" / "reps" / "tx_b.rep1.narrowPeak").write_text("grew since the dry run\n")
    (root / "peaks" / "tx_b" / "reps" / "tx_b.repT.xls").unlink()
    rows += "../outside.txt\t9\tforged\n"  # outside the root
    rows += "peaks/tx_a/tx_a.narrowPeak.gz\t2\tforged\n"  # what prep's check reads
    manifest.write_text(rows)
    c = prune.apply(p, manifest, stamp="U")
    assert c["changed"] == 1 and c["gone"] == 1 and c["refused"] == 2
    assert outside.read_text() == "precious\n" and not os.path.lexists(root / "peaks/tx_a/reps/link.narrowPeak")
    assert (root / "peaks" / "tx_a" / "tx_a.narrowPeak.gz").exists()
    assert (root / "peaks" / "tx_b" / "reps" / "tx_b.rep1.narrowPeak").exists()
    log = (root / "prune" / "deleted.U.tsv").read_text()
    assert "../outside.txt\t9\trefused" in log
    other = tmp_path / "other.tsv"
    other.write_text("# root\t/somewhere/else\npath\tbytes\trule\n")
    with pytest.raises(prune.PruneError, match="not of this project"):
        prune.apply(p, other)


def test_chrombpnet_and_variant_scoring_rulesets(tmp_path):
    shared = tmp_path / "shared.txt"
    shared.write_text("tx_a\n")
    p = pj.load(_yaml(tmp_path, prune={"rulesets": ["pipeline", "chrombpnet", "variant_scoring"],
                                       "shared": str(shared),
                                       "delete": [{"pattern": "variant_scoring/vendor/**", "why": "a clone"}]}))  # fmt: skip
    root = Path(p["root"])
    keep, delete = [], []
    for ds in p["datasets"]:
        fm = f"chrombpnet/{ds}/full_models/{ds}_all_fold_0"
        keep += [f"{fm}/models/chrombpnet_nobias.h5", f"{fm}/evaluation/chrombpnet_metrics.json",
                 f"{fm}/evaluation/chrombpnet_predictions.h5", f"{fm}/logs/chrombpnet.args.json",
                 f"{fm}/interpretation/interpretation.interpreted_regions.bed",
                 f"chrombpnet/{ds}/bias_models/b07/{ds}_all_fold_0/models/{ds}_all_fold_0_bias.h5",
                 f"chrombpnet/{ds}/bias_models/b07/{ds}_all_fold_0/evaluation/{ds}_all_fold_0_bias_metrics.json",
                 f"chrombpnet/{ds}/bias_models/b07/{ds}_all_fold_0/evaluation/motif_qc_counts/"
                 f"{ds}_all_fold_0_bias_modisco_results.h5",
                 f"chrombpnet/{ds}/full_models_bf16/{ds}_all_fold_0/evaluation/chrombpnet_metrics.json"]  # fmt: skip
        delete += [f"{fm}/interpretation/interpretation.profile_scores.h5", f"{fm}/interpretation/x.bw",
                   f"{fm}/evaluation/bias_predictions.h5", f"chrombpnet/{ds}/predictions/p.h5",
                   f"chrombpnet/{ds}/bias_models/b07/{ds}_all_fold_0/evaluation/{ds}_all_fold_0_bias_predictions.h5",
                   f"chrombpnet/{ds}/full_models_bf16/{ds}_all_fold_0/models/chrombpnet.h5"]  # fmt: skip
        sc = f"variant_scoring/run1/scores/{ds}"
        keep += [f"{sc}/cherimoya.variants.summary.tsv.gz", f"{sc}/cherimoya.variants.summary.npz"]
        delete += [f"{sc}/fold_0/cherimoya.variants.npz", f"variant_scoring/run1/preds/{ds}/p.npz"]
    counts = "full_models/{ds}_all_fold_0/interpretation/interpretation.counts_scores.h5"
    keep.append("chrombpnet/tx_a/" + counts.format(ds="tx_a"))  # shared: level 2 reads it
    delete.append("chrombpnet/tx_b/" + counts.format(ds="tx_b"))
    delete.append("variant_scoring/vendor/repo/setup.py")
    for rel in keep + delete:
        _put(root, rel)
    deleted = {rel for rel, _, _ in prune.plan(p).delete}
    assert deleted == set(delete)


@pytest.mark.parametrize(
    "block, message",
    [
        ({"surprise": 1}, "unknown keys"),
        ({"rulesets": ["nope"]}, "unknown rulesets"),
        ({"shared": ["tx_zz"]}, "not the project's"),
        ({"delete": ["../up/**"]}, "relative"),
    ],
)
def test_bad_prune_blocks(tmp_path, block, message):
    p = pj.load(_yaml(tmp_path, prune=block))
    with pytest.raises(prune.PruneError, match=message):
        prune.project_rules(p)


def test_driver_false_refuses_the_driver_commands(tmp_path, capsys):
    path = _yaml(tmp_path, driver=False)
    assert pj.main([str(path), "env"]) == pj.DRIVER_REFUSED
    assert "driver: false" in capsys.readouterr().err
    assert pj.main([str(path), "coop-inputs"]) == pj.DRIVER_REFUSED
    assert not (tmp_path / "root" / "cherimoya" / "cooperativity").exists()
    assert pj.main([str(path), "show"]) == 0 and pj.main([str(path), "check", "fetch"]) == 1
    assert pj.main([str(_yaml(tmp_path, reference_root=str(tmp_path / "refs"))), "env"]) == 0
    with pytest.raises(pj.ProjectError, match="true or false"):
        pj.load(_yaml(tmp_path, driver="no"))


def test_main_prune_dry_run_then_apply(tmp_path, capsys):
    path = _yaml(tmp_path)
    p = pj.load(path)
    tree = _project_tree(p)
    assert pj.main([str(path), "prune"]) == 0
    out = capsys.readouterr().out
    assert "dry run; nothing deleted" in out and f"would delete {len(tree['delete'])} files" in out
    manifest = next((Path(p["root"]) / "prune").glob("manifest.*.tsv"))
    assert all((Path(p["root"]) / rel).exists() for rel in tree["delete"])
    assert pj.main([str(path), "prune", "--apply", str(manifest)]) == 0
    assert not any((Path(p["root"]) / rel).exists() for rel in tree["delete"])


def test_share_packs_the_results_with_md5s(tmp_path, capsys):
    path = _yaml(tmp_path)
    p = pj.load(path)
    _project_tree(p)
    root = Path(p["root"])
    _put(root, "README.md", "# toy\n")
    (root / "cherimoya" / "tx_b" / "downstream" / "finemo_tx2" / "tx_b_all" / "hits_unique.tsv").unlink()
    files = share.share_set(p, ["tx_a", "tx_b"])
    assert root / "cherimoya" / "tx_a" / "fold_0" / "cherimoya.fold_0.torch" in files
    assert root / "cherimoya" / "tx_a" / "fold_0" / "attr.attr.npz" not in files  # the fold arrays stay behind
    out = tmp_path / "toy.tar"
    r = share.write_share(p, files, out)
    assert not (tmp_path / "toy.tar.partial").exists()
    assert any(m.endswith("tx_b_all/hits_unique.tsv") for m in r["missing"])
    with tarfile.open(out) as tar:
        names = tar.getnames()
        assert names[-2:] == ["toy/MANIFEST.md5", "toy/SHARE.txt"]
        manifest = tar.extractfile("toy/MANIFEST.md5").read().decode().splitlines()
        for line in manifest:
            md5, name = line.split("  ", 1)
            assert hashlib.md5(tar.extractfile(name).read()).hexdigest() == md5
        assert "toy/cherimoya/tx_a/downstream/finemo_tx2/tx_a_all/finemo_report/fig/a.png" in names
        assert "tx_b_all/hits_unique.tsv" in tar.extractfile("toy/SHARE.txt").read().decode()
    assert pj.main([str(path), "share", "--list", "--datasets", "tx_a"]) == 0
    out = capsys.readouterr().out
    assert "\ncherimoya/tx_a/fold_0/cherimoya.fold_0.torch\n" in out and "MISSING cherimoya/tx_a/fold_0/fit.json" in out
    assert pj.main([str(path), "share", "--datasets", "tx_zz", "--list"]) == 2
