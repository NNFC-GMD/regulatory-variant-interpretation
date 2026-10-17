#!/usr/bin/env python3
"""The tn5 pilot's decision: per arm, the pre-registered rule on TF-MoDISco patterns and test performance.

The models of one fold were attributed on the same peaks (ChromBPNet's 30K-peak QC subsample of the fold) and run
through TF-MoDISco at the QC scale (motif_qc.py), each with its head: Cherimoya's existing model (role baseline), the
arms trained with a Tn5 bias control track (arm), ChromBPNet corrected (chrombpnet_nobias.h5; corrected) and
uncorrected (chrombpnet.h5; uncorrected), and any model shown alongside without a verdict (reference). --spec names
them:

    {"models": [{"name": "<model>", "kind": "cherimoya"|"chrombpnet", "head": "profile"|"counts",
                 "role": "corrected"|"uncorrected"|"baseline"|"arm"|"reference",
                 "modisco": "<modisco-lite results h5>", "performance": "<compare.tsv>"|null,
                 "finetune": "<finetune.json>"|null}, ...],
     "motif_db": "<MEME file>", "bias_db": "<MEME file>", "thresholds": {"<name>": <value>, ...},
     "dataset": "<name>", "fold": "<k>"}  (dataset and fold optional, shown in decision.md's title)

A model appears at most once per head, and a head has at most one corrected, one uncorrected and one baseline model.
A model whose h5 is not there (an arm the alignment gate skipped) is reported as missing, not an error.
  motif_db  the database level 3 annotates with and searches as decoys (MotifCompendium's: the ref_db a compendium's
            compendium.json records; that compendium.json is accepted in its place)
  bias_db   the database with the TN5_/DNASE_ entries (chrombpnet's motifs.meme: the motif_db of ChromBPNet's
            <prefix>_motif_qc.json; that JSON is accepted in its place)

Matching is level 3's (compare/motifs.py; compare_modisco.annotate_patterns and compare_sets): tomtom-lite on trimmed
PPMs, the other set's patterns searched together with motif_db's motifs as decoys, BH q-values per query; a pattern
matches a set when its best target among that set's patterns has q <= 0.05 (seen from the query's side). Per head:
  uncorrected-only  the uncorrected model's patterns that match no pattern of the corrected model of its head
  tn5like           a pattern whose best bias_db match is a TN5_/DNASE_ entry at q <= 0.05 (bias_flag), or that
                    matches the uncorrected-only patterns (the uncorrected model's own: that is one). A head without
                    an uncorrected model (the counts head: contribs_bw refuses the uncorrected model's counts head)
                    takes another head's uncorrected-only patterns, as level 3's bias control does
  share             seqlets of the tn5like patterns / seqlets of all the model's patterns, pos and neg (each pattern's
                    n_seqlets; seqlets TF-MoDISco put in no pattern are not in its h5 and not counted); 0 without any
  TF matched        the corrected model's patterns that are not tn5like and match the model's patterns (the corrected
                    pattern is the query)
Performance is the model's row of its compare.tsv (compare_chrombpnet.py: `cherimoya` or `chrombpnet`; an arm's row
is its prediction with its real control, which compare_chrombpnet.py marks in a last column `control`; no column = no
control).

The decision, on the profile head, per arm (thresholds by name; defaults in THRESHOLDS, the spec's override them):
  T1  tn5like(arm) <= tn5like(corrected) + t1_extra_patterns  and  share(arm) <= share(corrected) + t1_share_margin
  T2  TF matched(arm) >= TF matched(baseline) - t2_margin
  T3  counts_pearson(arm) >= counts_pearson(baseline) - t3_pearson_margin
  T4  median_jsd(arm) <= median_jsd(baseline) + t4_jsd_margin
  guard  for the head-only fine-tune (an arm whose entry names a finetune record, or named headft): both norms of the
         control channel in the checkpoint it used, fconv.weight[:, n_filters, :] and linear.weight[:, n_filters]
         (finetune_head.py's finetune.json: checkpoint_used, control_channel_norms; see control_norms), are
         non-zero. A zero norm leaves the arm the baseline with an unused input: a no-op. A fine-tune that
         finetune.json records as failed fails the guard too.
PASS = T1-T4 and the guard pass; FAIL = one of them fails; undetermined = none fails but one cannot be computed (an
input is missing); missing = nothing of the arm is there. On one fold, PASS means worth retraining every fold, not a
final answer. The counts head gets the same numbers, without verdicts.

Writes into --out-dir, each file atomically: patterns.tsv (one row per pattern per model and head: level 3's
annotation and bias flag, the partners, tn5like), summary.tsv (one row per model and head), decision.json
(thresholds, definitions, inputs, numbers, verdicts) and decision.md (the tables; the time in ET). Exit 0 once they
are written, whatever the verdicts; 1 on a bad spec or an unreadable database.

    python workflows/cherimoya/tn5/compare_arms.py --spec <dir>/spec.json --out-dir <dir> [--threads N]
    python workflows/cherimoya/tn5/compare_arms.py --selftest

Runs in the pixi cherimoya environment (h5py, memelite, pandas), on CPU; tomtom-lite runs on --threads threads
(NUMBA_NUM_THREADS; default the environment's, else 1). --selftest writes synthetic TF-MoDISco results, databases,
compare.tsv and finetune.json files into a temporary directory and checks the rule on them, first with an exact
stand-in for tomtom-lite (every match known, so verdicts and numbers are checked exactly), then with tomtom-lite
itself through the CLI (checking only what identical copies guarantee): no network, no real inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "compare"), str(HERE.parent)]

import common  # noqa: E402
import compare_modisco as cm  # noqa: E402
import motifs  # noqa: E402

logger = logging.getLogger("compare_arms")

FORMAT = 1
KINDS = ("cherimoya", "chrombpnet")
HEADS = ("profile", "counts")
ROLES = ("corrected", "uncorrected", "baseline", "arm", "reference")
DECISION_HEAD = "profile"
"""The head the rule decides on; the counts head is reported alongside, without verdicts."""
THRESHOLDS = {"q": motifs.Q_MATCH, "t1_extra_patterns": 1, "t1_share_margin": 0.02, "t2_margin": 1,
              "t3_pearson_margin": 0.01, "t4_jsd_margin": 0.001}  # fmt: skip
"""The pre-registered rule's numbers. q is level 3's (motifs.Q_MATCH, fixed in its matching and flags)."""
GUARDED = ("headft",)
"""Arms the control-channel guard checks besides any arm whose entry names a finetune record."""
PERF_KEYS = ("n_regions", "counts_pearson", "counts_spearman", "counts_mse", "median_jsd", "median_norm_jsd")
CKPT_KEYS = ("checkpoint_used", "checkpoint", "selected_checkpoint", "chosen_checkpoint")
NORM_KEYS = ("control_channel_norms", "control_norms", "norms")
"""Where finetune.json keeps the checkpoint used and the control-channel norms: finetune_head.py's keys first."""
EPS = 1e-12
"""Slack of the threshold comparisons (a limit computed from values printed with 6 significant digits)."""
ET = ZoneInfo("America/New_York")


@dataclass
class Model:
    """One model and head of the spec, with its patterns once read."""

    name: str
    kind: str
    head: str
    role: str
    modisco: Path | None
    performance: Path | None
    finetune: Path | None
    state: str = "missing"
    """ok, missing (no h5 named or none there) or unreadable"""
    note: str = ""
    patterns: list = field(default_factory=list)
    rows: dict = field(default_factory=dict)
    """{pattern key: its patterns.tsv row}"""
    tf_matched: list | None = None
    """The keys of the corrected model's TF patterns that match this model's; None without a corrected model."""

    @property
    def label(self) -> str:
        return f"{self.name}.{self.head}"

    @property
    def ok(self) -> bool:
        return self.state == "ok"


# ---------------------------------------------------------------------------------------------------------------------
# inputs


def read_spec(path: Path) -> tuple[list[Model], dict, dict, dict]:
    """The models, the thresholds (THRESHOLDS updated by the spec's), the two databases as given and the context
    (dataset and fold, when the spec names them)."""
    spec = json.loads(Path(path).read_text())
    if not isinstance(spec, dict) or not isinstance(spec.get("models"), list) or not spec["models"]:
        raise ValueError(f"{path}: no models")
    models, seen = [], set()
    for i, e in enumerate(spec["models"]):
        if not isinstance(e, dict) or not isinstance(e.get("name"), str) or not e["name"]:
            raise ValueError(f"{path}: models[{i}] has no name")
        for key, allowed in (("kind", KINDS), ("head", HEADS), ("role", ROLES)):
            if e.get(key) not in allowed:
                raise ValueError(f"{path}: {e['name']}: {key} {e.get(key)!r} is not one of {', '.join(allowed)}")
        if (e["name"], e["head"]) in seen:
            raise ValueError(f"{path}: {e['name']} appears twice for the {e['head']} head")
        seen.add((e["name"], e["head"]))
        paths = {k: Path(e[k]) if e.get(k) else None for k in ("modisco", "performance", "finetune")}
        models.append(Model(e["name"], e["kind"], e["head"], e["role"], **paths))
    for head in HEADS:
        for role in ("corrected", "uncorrected", "baseline"):
            n = sum(m.head == head and m.role == role for m in models)
            if n > 1:
                raise ValueError(f"{path}: {n} {role} models for the {head} head; the rule takes one")
    given = spec.get("thresholds") or {}
    unknown = sorted(set(given) - set(THRESHOLDS))
    if unknown:
        raise ValueError(f"{path}: unknown thresholds {', '.join(unknown)} (known: {', '.join(THRESHOLDS)})")
    for k, v in given.items():
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
            raise ValueError(f"{path}: threshold {k} = {v!r} is not a number")
    thresholds = {**THRESHOLDS, **given}
    if thresholds["q"] != motifs.Q_MATCH:
        raise ValueError(f"{path}: q = {thresholds['q']}, but level 3's matching and flags use {motifs.Q_MATCH}")
    dbs = {k: spec.get(k) for k in ("motif_db", "bias_db")}
    missing = [k for k, v in dbs.items() if not v]
    if missing:
        raise ValueError(f"{path}: no {' and no '.join(missing)}")
    context = {k: str(spec[k]) for k in ("dataset", "fold") if spec.get(k) is not None}
    return models, thresholds, dbs, context


def resolve_db(given: str, key: str) -> Path:
    """A MEME file, or the JSON that records one: a compendium.json (ref_db.filepath, the database 09.0 annotated the
    compendium with) or a motif_qc.json (motif_db, the database motif_qc.py reported against)."""
    path = Path(given)
    if path.suffix == ".json":
        rec = json.loads(path.read_text())
        ref = rec["ref_db"].get("filepath") if isinstance(rec.get("ref_db"), dict) else rec.get("motif_db")
        if not isinstance(ref, str) or not ref:
            raise ValueError(f"{key} {path}: records no MEME file (ref_db.filepath or motif_db)")
        path = Path(ref)
    if not path.is_file():
        raise ValueError(f"{key} {path}: no such file")
    return path


def load_patterns(models: list[Model]) -> None:
    """Each model's patterns (motifs.read_modisco, without the seqlet arrays), or its state when there are none: missing
    (no h5 named, or none there) or unreadable (e.g. an h5 TF-MoDISco did not finish writing)."""
    for m in models:
        if m.modisco is None or not m.modisco.is_file():
            m.state, m.note = "missing", f"no TF-MoDISco results at {m.modisco}" if m.modisco else "no h5 named"
            continue
        try:
            m.patterns, _ = motifs.read_modisco(m.modisco, m.name, seqlets=False)
        except (OSError, KeyError, ValueError) as exc:
            m.state, m.note = "unreadable", f"{m.modisco}: {exc}"
            continue
        m.state = "ok"


def read_performance(path: Path, kind: str) -> dict:
    """The model's row of a compare.tsv (one row per model, `cherimoya` and `chrombpnet`; columns by name, so an
    arm's control column changes nothing). Of two rows of the model, the one whose control is real."""
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t") if r.get("model") == kind]
    if len(rows) > 1:
        rows = [r for r in rows if (r.get("control") or "").strip() == "real"]
    if len(rows) != 1:
        raise ValueError(f"{path}: no single {kind} row")
    out = {}
    for k in PERF_KEYS:
        try:
            out[k] = float(rows[0][k])
        except (KeyError, TypeError, ValueError):
            out[k] = math.nan
    if not (math.isfinite(out["counts_pearson"]) and math.isfinite(out["median_jsd"])):
        raise ValueError(f"{path}: the {kind} row has no counts_pearson or median_jsd")
    out["control"] = (rows[0].get("control") or "").strip() or "none"
    return out


def performances(models: list[Model]) -> dict[str, dict]:
    """{model name: {"perf": dict or None, "path", "note"}}, from the first of its entries that names a compare.tsv."""
    out: dict[str, dict] = {}
    for m in models:
        if m.name in out and out[m.name]["path"] is not None:
            continue
        entry = {"perf": None, "path": m.performance, "note": "no compare.tsv named"}
        if m.performance is not None and not m.performance.is_file():
            entry["note"] = f"no {m.performance}"
        elif m.performance is not None:
            try:
                entry.update(perf=read_performance(m.performance, m.kind), note="")
            except (OSError, ValueError) as exc:
                entry["note"] = str(exc)
        out[m.name] = entry
    return out


def checkpoint_tag(text: str) -> str:
    """.final.torch (the last epoch) or .torch (the best validation epoch, the normal fits' choice), by the file name
    (or key): final or last in it means the former."""
    name = Path(str(text)).name.lower()
    return ".final.torch" if "final" in name or "last" in name else ".torch"


def control_norms(record: dict) -> tuple[str | None, dict | None, str]:
    """(the checkpoint used, {"fconv": norm, "linear": norm} of its control channel, a note when they are not found)
    from finetune_head.py's finetune.json: the checkpoint under one of CKPT_KEYS (default .torch); the norms under one
    of NORM_KEYS, either per checkpoint ({".torch": {...}, ".final.torch": {...}}, keys told apart by checkpoint_tag)
    or of the one used; in them, the value under fconv and linear, else the one key naming each."""
    ckpt = next((record[k] for k in CKPT_KEYS if isinstance(record.get(k), str)), None)
    tag = checkpoint_tag(ckpt) if ckpt else ".torch"
    box = next((record[k] for k in NORM_KEYS if isinstance(record.get(k), dict)), None)
    if box is None:
        return ckpt, None, f"no control-channel norms ({' / '.join(NORM_KEYS)}) among {', '.join(sorted(record))}"
    entry = box
    if any(isinstance(v, dict) for v in box.values()):
        entries = {checkpoint_tag(k): v for k, v in box.items() if isinstance(v, dict)}
        if tag not in entries:
            return ckpt, None, f"no norms of the {tag} checkpoint among {', '.join(box)}"
        entry = entries[tag]

    def number(v) -> bool:
        return isinstance(v, int | float) and not isinstance(v, bool)

    norms = {}
    for part in ("fconv", "linear"):
        vals = [entry[part]] if number(entry.get(part)) else [v for k, v in entry.items()
                                                                 if part in str(k).lower() and number(v)]  # fmt: skip
        if len(vals) != 1:
            return ckpt, None, f"{len(vals)} {part} norms among {', '.join(map(str, entry))}"
        norms[part] = float(vals[0])
    return ckpt or tag, norms, ""


def guard(path: Path | None) -> dict:
    """The head-only fine-tune's guard: both control-channel norms of the checkpoint it used are non-zero."""
    out = {"finetune": str(path) if path else None, "checkpoint": None, "fconv_norm": None, "linear_norm": None,
           "pass": None, "note": ""}  # fmt: skip
    if path is None or not path.is_file():
        out["note"] = f"no finetune.json at {path}" if path else "no finetune.json named"
        return out
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        out["note"] = f"unreadable {path}: {exc}"
        return out
    record = record if isinstance(record, dict) else {}
    ckpt, norms, note = control_norms(record)
    out.update(checkpoint=ckpt, note=note, status=record.get("status"))
    if record.get("status") == "failed":
        out["pass"], out["note"] = False, f"the fine-tune failed: {record.get('reason') or 'no reason recorded'}"
    if norms is None:
        return out
    out.update(fconv_norm=norms["fconv"], linear_norm=norms["linear"])
    zero = [k for k, v in norms.items() if not (math.isfinite(v) and v > 0)]
    if zero:
        out["pass"], out["note"] = False, f"zero {' and '.join(zero)} control-channel norm in {ckpt}: a no-op"
    elif out["pass"] is None:
        out["pass"] = True
    return out


# ---------------------------------------------------------------------------------------------------------------------
# patterns


def pair(a: list[motifs.Pattern], b: list[motifs.Pattern], decoys: motifs.Database) -> dict:
    """Level 3's matching of two pattern sets both ways, without seqlets (compare_modisco.compare_sets): {"a": {key:
    entry}, "b": {key: entry}}, entry["tomtom_partner"] the key of the best target among the other set's patterns
    when its q <= motifs.Q_MATCH (else None), with its tomtom_p and tomtom_q."""
    return cm.compare_sets({"a": a, "b": b}, None, None, decoys, seqlet_overlap=False)["per"]


def find(models: list[Model], role: str, head: str) -> Model | None:
    """The head's model of that role, when it is there and readable."""
    m = next((m for m in models if m.role == role and m.head == head), None)
    return m if m is not None and m.ok else None


def evaluate(models: list[Model], annot_db: motifs.Database, bias_db: motifs.Database) -> dict:
    """Annotates every pattern, finds the uncorrected-only patterns, marks tn5like and the corrected model's TF
    patterns each model matches. Returns {head: {"corrected", "baseline", "uncorrected" (a Model or None),
    "uncorrected_only" (keys or None), "note"}}."""
    present = [m for m in models if m.ok]
    for m in present:
        m.rows = {r["pattern"]: r for r in cm.annotate_patterns(m.patterns, annot_db, bias_db)}
        for r in m.rows.values():
            r.update(head=m.head, kind=m.kind, role=m.role)
    uonly: dict[str, list[str] | None] = {}
    for u in (m for m in present if m.role == "uncorrected"):
        c = find(models, "corrected", u.head)
        if c is None:
            uonly[u.label] = None
            continue
        per = pair(u.patterns, c.patterns, annot_db)["a"]
        uonly[u.label] = [p.key for p in u.patterns if per[p.key]["tomtom_partner"] is None]
    refs = {}
    for head in dict.fromkeys(m.head for m in models):
        unc, note = find(models, "uncorrected", head), ""
        if unc is None or uonly.get(unc.label) is None:
            why = (f"no readable {head}-head uncorrected model" if unc is None
                   else f"no readable corrected model of the {head} head for {unc.label}")  # fmt: skip
            others = sorted((m for m in present if m.role == "uncorrected" and uonly.get(m.label) is not None),
                            key=lambda m: m.head != DECISION_HEAD)  # fmt: skip
            unc = others[0] if others else None
            note = (f"{why}: {unc.label}'s uncorrected-only patterns stand in" if unc is not None
                    else f"{why}: tn5like is the bias_db flag alone")  # fmt: skip
        refs[head] = {"corrected": find(models, "corrected", head), "baseline": find(models, "baseline", head),
                      "uncorrected": unc, "uncorrected_only": uonly.get(unc.label) if unc else None, "note": note}  # fmt: skip
    for m in present:
        mark_tn5like(m, refs[m.head], annot_db)
    for m in present:
        match_corrected(m, refs[m.head]["corrected"], annot_db)
    return refs


def mark_tn5like(m: Model, ref: dict, decoys: motifs.Database) -> None:
    """tn5like of each pattern of m: its bias flag, or a match among the uncorrected-only patterns; with its partner
    among all the uncorrected model's patterns, for the table."""
    unc, keys = ref["uncorrected"], set(ref["uncorrected_only"] or [])
    to_sub, to_all = {}, {}
    if unc is not None and unc is not m:
        to_sub = pair(m.patterns, [p for p in unc.patterns if p.key in keys], decoys)["a"]
        to_all = pair(m.patterns, unc.patterns, decoys)["a"]
    for p in m.patterns:
        r = m.rows[p.key]
        r.update(uncorrected_reference=unc.label if unc is not None else None, uncorrected_partner=None,
                 uncorrected_q=np.nan, uncorrected_only=False, uncorrected_only_partner=None,
                 uncorrected_only_q=np.nan)  # fmt: skip
        if unc is m:
            r.update(uncorrected_partner=p.key, uncorrected_only=p.key in keys,
                     uncorrected_only_partner=p.key if p.key in keys else None)  # fmt: skip
        elif unc is not None:
            e, s = to_all[p.key], to_sub[p.key]
            r.update(uncorrected_partner=e["tomtom_partner"], uncorrected_q=e["tomtom_q"],
                     uncorrected_only=s["tomtom_partner"] is not None, uncorrected_only_partner=s["tomtom_partner"],
                     uncorrected_only_q=s["tomtom_q"])  # fmt: skip
        by = [w for w, on in (("bias_db", bool(r["bias_flag"])), ("uncorrected-only", r["uncorrected_only"])) if on]
        r["tn5like"], r["tn5like_by"] = bool(by), "+".join(by)


def match_corrected(m: Model, c: Model | None, decoys: motifs.Database) -> None:
    """Each pattern of m's partner among the corrected model's patterns, and the corrected TF patterns (not tn5like)
    that find a partner among m's (the corrected pattern as the query)."""
    for r in m.rows.values():
        r.update(corrected_reference=c.label if c is not None else None, corrected_partner=None, corrected_q=np.nan)
    if c is None:
        return
    tf = [p for p in c.patterns if not c.rows[p.key]["tn5like"]]
    if m is c:
        for p in m.patterns:
            m.rows[p.key]["corrected_partner"] = p.key
        m.tf_matched = [p.key for p in tf]
        return
    per = pair(c.patterns, m.patterns, decoys)
    for p in m.patterns:
        e = per["b"][p.key]
        m.rows[p.key].update(corrected_partner=e["tomtom_partner"], corrected_q=e["tomtom_q"])
    m.tf_matched = [p.key for p in tf if per["a"][p.key]["tomtom_partner"] is not None]
    for key in m.tf_matched:
        found = c.rows[key].get("tf_found_in")
        c.rows[key]["tf_found_in"] = f"{found},{m.name}" if found else m.name


# ---------------------------------------------------------------------------------------------------------------------
# numbers and verdicts


def model_numbers(m: Model, ref: dict) -> dict:
    """A model's summary.tsv row: pattern and seqlet counts, tn5like count and share, TF matched."""
    out = {"model": m.name, "head": m.head, "kind": m.kind, "role": m.role, "state": m.state, "note": m.note,
           "modisco": str(m.modisco) if m.modisco else None}  # fmt: skip
    if not m.ok:
        return out
    seqlets = sum(p.n_seqlets for p in m.patterns)
    tn5 = sum(p.n_seqlets for p in m.patterns if m.rows[p.key]["tn5like"])
    c, unc = ref["corrected"], ref["uncorrected"]
    partnered = sum(r["corrected_partner"] is not None for r in m.rows.values())
    out.update(
        patterns_pos=sum(p.posneg == "pos" for p in m.patterns),
        patterns_neg=sum(p.posneg == "neg" for p in m.patterns),
        seqlets=seqlets, bias_flagged=sum(bool(r["bias_flag"]) for r in m.rows.values()),
        uncorrected_only=sum(bool(r["uncorrected_only"]) for r in m.rows.values()),
        tn5like=sum(bool(r["tn5like"]) for r in m.rows.values()), tn5like_seqlets=tn5,
        tn5like_share=tn5 / seqlets if seqlets else 0.0,
        uncorrected_reference=unc.label if unc is not None else None,
        corrected_reference=c.label if c is not None else None,
        corrected_tf=sum(not r["tn5like"] for r in c.rows.values()) if c is not None else None,
        tf_matched=len(m.tf_matched) if m.tf_matched is not None else None,
        matched_in_corrected=partnered if c is not None else None,
    )  # fmt: skip
    return out


def le(a: float, b: float) -> bool:
    return a <= b + EPS


def arm_tests(name: str, head: str, models: list[Model], refs: dict, numbers: dict, perf: dict, thresholds: dict,
              verdicts: bool) -> dict:  # fmt: skip
    """T1-T4 (and the guard) of one arm on one head: the numbers and limits and, with verdicts, pass (None when an
    input is missing), a note and the verdict with its reasons."""
    th = thresholds
    entry = next((m for m in models if m.name == name and m.head == head), None)
    ref = refs.get(head) or {"corrected": None, "baseline": None}
    arm = numbers[(name, head)] if entry is not None and entry.ok else None
    cor = numbers[(ref["corrected"].name, head)] if ref["corrected"] is not None else None
    base = numbers[(ref["baseline"].name, head)] if ref["baseline"] is not None else None
    if arm is None:
        lacking = f"no {head}-head entry" if entry is None else f"{entry.label} {entry.state}: {entry.note}"
    else:
        lacking = f"no readable corrected model of the {head} head" if cor is None else ""
    tests = {}
    t = {"n_tn5like": None, "n_limit": None, "n_pass": None, "share": None, "share_limit": None, "share_pass": None,
         "pass": None, "note": lacking}  # fmt: skip
    if arm is not None and cor is not None:
        t.update(n_tn5like=arm["tn5like"], n_limit=cor["tn5like"] + th["t1_extra_patterns"], share=arm["tn5like_share"],
                 share_limit=cor["tn5like_share"] + th["t1_share_margin"])  # fmt: skip
        t.update(n_pass=le(t["n_tn5like"], t["n_limit"]), share_pass=le(t["share"], t["share_limit"]))
        t["pass"] = t["n_pass"] and t["share_pass"]
        t["note"] = " and ".join(w for w, ok in (("more tn5like patterns", t["n_pass"]),
                                                 ("a larger tn5like share", t["share_pass"])) if not ok)  # fmt: skip
    tests["T1"] = t
    t = {"matched": None, "of": None, "baseline_matched": None, "limit": None, "pass": None, "note": lacking}
    if arm is not None and cor is not None and base is None:
        t["note"] = f"no readable baseline model of the {head} head"
    elif arm is not None and cor is not None:
        t.update(matched=arm["tf_matched"], of=cor["corrected_tf"], baseline_matched=base["tf_matched"],
                 limit=base["tf_matched"] - th["t2_margin"])  # fmt: skip
        t["pass"] = le(t["limit"], t["matched"])
        t["note"] = "" if t["pass"] else "fewer of the corrected model's TF patterns matched"
    tests["T2"] = t
    base_name = ref["baseline"].name if ref["baseline"] is not None else next(
        (m.name for m in models if m.role == "baseline"), None)  # fmt: skip
    p_arm = perf.get(name, {}).get("perf")
    p_base = perf.get(base_name, {}).get("perf") if base_name else None
    for key, measure, sign, margin in (("T3", "counts_pearson", -1, "t3_pearson_margin"),
                                       ("T4", "median_jsd", +1, "t4_jsd_margin")):  # fmt: skip
        t = {"value": None, "baseline": None, "limit": None, "pass": None, "note": ""}
        if p_arm is not None and p_base is not None:
            t.update(value=p_arm[measure], baseline=p_base[measure], limit=p_base[measure] + sign * th[margin])
            t["pass"] = le(t["value"], t["limit"]) if sign > 0 else le(t["limit"], t["value"])
            t["note"] = "" if t["pass"] else f"{measure} {'above' if sign > 0 else 'below'} the limit"
        elif p_arm is None:
            t["note"] = f"{name}: {perf.get(name, {}).get('note') or 'no entry'}"
        else:
            t["note"] = f"baseline {base_name}: {perf[base_name]['note']}" if base_name else "no baseline model"
        tests[key] = t
    out = {"model": name, "head": head, "state": entry.state if entry is not None else "missing", "tests": tests}
    if p_arm is not None:
        out["control"] = p_arm["control"]
        if p_base is not None and math.isfinite(p_arm["n_regions"] - p_base["n_regions"]) and (
                p_arm["n_regions"] != p_base["n_regions"]):  # fmt: skip
            out["performance_note"] = (f"{p_arm['n_regions']:.0f} test regions against the baseline's "
                                       f"{p_base['n_regions']:.0f}: not the same regions")  # fmt: skip
    finetune = next((m.finetune for m in models if m.name == name and m.finetune is not None), None)
    if not verdicts:
        for t in tests.values():
            if t["pass"] is not None:
                t.pop("note")
            for k in [k for k in t if k == "pass" or k.endswith("_pass")]:
                t.pop(k)
        return out
    if finetune is not None or name in GUARDED:
        out["guard"] = guard(finetune)
    checks = {**tests, **({"guard": out["guard"]} if "guard" in out else {})}
    failed = [k for k, t in checks.items() if t["pass"] is False]
    unknown = [k for k, t in checks.items() if t["pass"] is None]
    if arm is None and p_arm is None and (finetune is None or not finetune.is_file()):
        out["verdict"], out["reasons"] = "missing", []
        out["note"] = "none of its TF-MoDISco results, compare.tsv or finetune.json is there (skipped by the gate?)"
    elif failed:
        out["verdict"], out["reasons"] = "FAIL", [f"{k}: {checks[k]['note']}" for k in failed]
    elif unknown:
        out["verdict"], out["reasons"] = "undetermined", [f"{k}: {checks[k]['note']}" for k in unknown]
    else:
        out["verdict"], out["reasons"] = "PASS", []
    return out


# ---------------------------------------------------------------------------------------------------------------------
# outputs


PATTERN_FRONT = ["model", "head", "kind", "role", "posneg", "pattern", "n_seqlets"]
DEFINITIONS = {
    "matching": "tomtom-lite on trimmed PPMs (level 3), the other set's patterns searched with motif_db's motifs as "
    "decoys, BH q per query; a pattern matches a set when its best target among that set's patterns has q <= 0.05",
    "uncorrected_only": "the uncorrected model's patterns that match no pattern of the corrected model of its head",
    "tn5like": "best bias_db match a TN5_/DNASE_ entry at q <= 0.05 (bias_flag), or a match among the uncorrected-only "
    "patterns (for the uncorrected model's own patterns: being one)",
    "seqlet_share": "seqlets of the tn5like patterns / seqlets of all the model's patterns, pos and neg (n_seqlets per "
    "pattern; seqlets in no pattern are not counted); 0 when the model has no pattern",
    "tf_matched": "the corrected model's patterns that are not tn5like and match the model's patterns (the corrected "
    "pattern is the query)",
    "performance": "the model's row of its compare.tsv (cherimoya or chrombpnet); an arm's row is predicted with its "
    "control",
}


def patterns_table(models: list[Model]):
    import pandas as pd

    rows = [m.rows[p.key] for m in models if m.ok for p in m.patterns]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=PATTERN_FRONT)
    return df[PATTERN_FRONT + [c for c in df.columns if c not in PATTERN_FRONT]]


def fmt(v, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    if isinstance(v, int | np.integer) and not isinstance(v, bool):
        return str(int(v))
    if digits == 0 and float(v).is_integer():
        return str(int(v))
    return f"{v:.{digits}f}"


def cell(value, limit, op: str, ok, digits: int) -> str:
    if value is None:
        return "-"
    mark = "" if ok is None else " ok" if ok else " **miss**"
    return f"{fmt(value, digits)} ({op} {fmt(limit, digits)}){mark}"


def guard_cell(g: dict | None) -> str:
    if g is None:
        return "-"
    if g["fconv_norm"] is None:
        return "n/a"
    return f"fconv {g['fconv_norm']:.3g}, linear {g['linear_norm']:.3g}" + (" ok" if g["pass"] else " **miss**")


def decision_md(decision: dict, numbers: dict) -> str:
    """The tables: the profile head's arms with verdicts and the other models' numbers, then the counts head's."""
    th = decision["thresholds"]
    lines = [
        "# tn5 pilot: decision" + "".join(f", {k} {v}" for k, v in decision["context"].items()),
        "",
        f"Written {decision['written_et']} from `{decision['spec']}`.",
        "",
        "Pre-registered rule, on the profile head: PASS = T1-T4 pass (and, for the head-only fine-tune, the guard). "
        "On one fold, PASS means worth retraining every fold, not a final answer.",
        f"T1 tn5like patterns <= corrected + {th['t1_extra_patterns']} and tn5like seqlet share <= corrected + "
        f"{th['t1_share_margin']}; T2 corrected TF patterns matched >= baseline - {th['t2_margin']}; T3 counts "
        f"Pearson >= baseline - {th['t3_pearson_margin']}; T4 median JSD <= baseline + {th['t4_jsd_margin']}; "
        f"matches at q <= {th['q']}.",
        "",
        "## Profile head (decides)",
        "",
        "| model | role | verdict | T1 tn5like | T1 share | T2 TF matched | T3 counts r | T4 median JSD | guard "
        "| why |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in decision["heads"].get(DECISION_HEAD, {}).get("arms", []):
        t1, t2, t3, t4 = (a["tests"][k] for k in ("T1", "T2", "T3", "T4"))
        t2txt = "-" if t2["matched"] is None else (f"{t2['matched']}/{t2['of']} (>= {fmt(t2['limit'], 0)})"
                                                   + (" ok" if t2["pass"] else " **miss**"))  # fmt: skip
        why = "; ".join(a["reasons"] + [a[k] for k in ("note", "performance_note") if a.get(k)]) or "-"
        cells = [a["model"], "arm", f"**{a['verdict']}**",
                 cell(t1["n_tn5like"], t1["n_limit"], "<=", t1["n_pass"], 0),
                 cell(t1["share"], t1["share_limit"], "<=", t1["share_pass"], 3), t2txt,
                 cell(t3["value"], t3["limit"], ">=", t3["pass"], 4),
                 cell(t4["value"], t4["limit"], "<=", t4["pass"], 4), guard_cell(a.get("guard")), why]  # fmt: skip
        lines.append("| " + " | ".join(cells) + " |")
    lines += reference_rows(numbers, DECISION_HEAD, wide=True)
    lines += ["", "## Counts head (secondary, no verdicts)", "",
              "| model | role | tn5like | share | TF matched | counts r | median JSD |",
              "|---|---|---|---|---|---|---|"]  # fmt: skip
    for a in decision["heads"].get("counts", {}).get("arms", []):
        t1, t2, t3, t4 = (a["tests"][k] for k in ("T1", "T2", "T3", "T4"))
        tf = "-" if t2["matched"] is None else f"{t2['matched']}/{t2['of']}"
        role = "arm" if a["state"] == "ok" else f"arm ({a['state']})"
        cells = [a["model"], role, fmt(t1["n_tn5like"], 0), fmt(t1["share"], 3), tf, fmt(t3["value"]), fmt(t4["value"])]
        lines.append("| " + " | ".join(cells) + " |")
    lines += reference_rows(numbers, "counts", wide=False)
    lines += ["", "## Notes", ""]
    for head, h in decision["heads"].items():
        r = h["references"]
        lines.append(f"- {head} head: corrected {r['corrected'] or 'none'}, baseline {r['baseline'] or 'none'}; "
                     f"uncorrected-only patterns of {r['uncorrected'] or 'none'}: "
                     f"{', '.join(r['uncorrected_only'] or []) or 'none'}" + (f" ({r['note']})" if r["note"] else ""))  # fmt: skip
    lines += [f"- {k}: {d}" for k, d in decision["definitions"].items()]
    lines += [f"- {n}" for n in decision["notes"]]
    lines.append(f"- motif_db `{decision['databases']['motif_db']}`, bias_db `{decision['databases']['bias_db']}`")
    return "\n".join(lines) + "\n"


def reference_rows(numbers: dict, head: str, wide: bool) -> list[str]:
    """The head's models other than the arms, with their numbers and no verdict."""
    out = []
    for (name, h), n in numbers.items():
        if h != head or n["role"] == "arm":
            continue
        tf = "-" if n.get("tf_matched") is None else f"{n['tf_matched']}/{n['corrected_tf']}"
        role = n["role"] if n["state"] == "ok" else f"{n['role']} ({n['state']})"
        cells = [name, role, fmt(n.get("tn5like"), 0), fmt(n.get("tn5like_share"), 3), tf,
                 fmt(n.get("counts_pearson")), fmt(n.get("median_jsd"))]  # fmt: skip
        if wide:
            cells = cells[:2] + [""] + cells[2:] + ["", n["note"] or ""]
        out.append("| " + " | ".join(cells) + " |")
    return out


def run(spec_path: Path, out_dir: Path) -> str:
    """Reads the spec and the patterns, decides, writes the four outputs; returns the one-line note."""
    import pandas as pd

    models, thresholds, given, context = read_spec(spec_path)
    dbs = {k: resolve_db(v, k) for k, v in given.items()}
    annot_db, bias_db = motifs.Database.read(dbs["motif_db"]), motifs.Database.read(dbs["bias_db"])
    load_patterns(models)
    for m in models:
        if not m.ok:
            logger.warning("%s: %s (%s)", m.label, m.state, m.note)
    refs = evaluate(models, annot_db, bias_db)
    perf = performances(models)
    numbers = {(m.name, m.head): model_numbers(m, refs[m.head]) for m in models}
    for (name, _), n in numbers.items():
        p = perf[name]["perf"] or {}
        n.update({k: p.get(k) for k in PERF_KEYS}, performance_control=p.get("control"),
                 performance_note=perf[name]["note"] or None)  # fmt: skip
    arms = list(dict.fromkeys(m.name for m in models if m.role == "arm"))
    heads = {}
    for head in (h for h in HEADS if h in refs):
        r = refs[head]
        labels = {k: r[k].label if r[k] is not None else None for k in ("corrected", "baseline", "uncorrected")}
        heads[head] = {
            "verdicts": head == DECISION_HEAD,
            "references": {**labels, "uncorrected_only": r["uncorrected_only"], "note": r["note"]},
            "arms": [arm_tests(a, head, models, refs, numbers, perf, thresholds, head == DECISION_HEAD) for a in arms],
        }
    decided = heads.get(DECISION_HEAD, {}).get("arms", [])
    notes = [f"{a['model']}: its compare.tsv row records control {a['control']!r}, not the real control"
             for a in decided if a.get("control") not in (None, "real")]  # fmt: skip
    if any(a["tests"]["T3"]["pass"] is False for a in decided):
        notes.append("a T3 miss of an arm trained with a control may come from the counts head's log(sum(control) + 1) "
                     "input rather than from the profile correction")  # fmt: skip
    inputs = {"spec": spec_path, **{f"db/{k}": v for k, v in dbs.items()}}
    for m in models:
        # By model and head: a head whose entry names no compare.tsv or finetune.json leaves the other head's in.
        inputs.update({f"{m.label}/modisco": m.modisco, f"{m.label}/performance": m.performance,
                       f"{m.label}/finetune": m.finetune})  # fmt: skip
    decision = {
        "format": FORMAT,
        "written_et": datetime.now(ET).strftime("%Y-%m-%d %H:%M %Z"),
        "spec": str(spec_path),
        "context": context,
        "thresholds": thresholds,
        "definitions": DEFINITIONS,
        "databases": {k: str(v) for k, v in dbs.items()},
        "databases_given": given,
        "heads": heads,
        "performance": {n: {**e, "path": str(e["path"]) if e["path"] else None} for n, e in perf.items()},
        "models": {f"{name}.{head}": n for (name, head), n in numbers.items()},
        "notes": notes,
        "inputs": common.fingerprint(inputs),
    }
    out_dir = Path(out_dir)
    common.write_table(out_dir / "patterns.tsv", patterns_table(models))
    common.write_table(out_dir / "summary.tsv", pd.DataFrame(list(numbers.values())))
    common.write_json(out_dir / "decision.json", decision)
    common.write_text(out_dir / "decision.md", decision_md(decision, numbers))
    said = []
    for a in decided:
        why = ",".join(r.split(":")[0] for r in a["reasons"])
        said.append(f"{a['model']} {a['verdict']}" + (f" ({why})" if why else ""))
    return (f"{DECISION_HEAD} head: {', '.join(said) or 'no arm'}; {len(models)} model entries, "
            f"{sum(not m.ok for m in models)} without patterns; wrote {out_dir}/decision.md")  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------------
# selftest


def _identity_tomtom(queries: list[np.ndarray], targets: list[np.ndarray]) -> np.ndarray:
    """Stand-in for motifs.tomtom_p: p = 1e-12 for a target equal to the query on either strand, else 0.5."""
    p = np.full((len(queries), len(targets)), 0.5)
    for i, q in enumerate(queries):
        for j, t in enumerate(targets):
            if q.shape == t.shape and (np.allclose(q, t, atol=1e-5) or np.allclose(q, t[::-1, ::-1], atol=1e-5)):
                p[i, j] = 1e-12
    return p


def _write_meme(path: Path, mats: dict[str, np.ndarray]) -> Path:
    lines = ["MEME version 4", "", "ALPHABET= ACGT", "", "strands: + -", "", "Background letter frequencies:",
             "A 0.25 C 0.25 G 0.25 T 0.25", ""]  # fmt: skip
    for name, ppm in mats.items():
        lines += [f"MOTIF {name}", f"letter-probability matrix: alength= 4 w= {len(ppm)} nsites= 20 E= 0"]
        lines += [" ".join(f"{v:.6f}" for v in row) for row in ppm] + [""]
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_modisco(path: Path, items: list[tuple[str, np.ndarray, int]], window: int = 400) -> list[str]:
    """A modisco-lite results h5 of items (posneg, (w, 4) PPM, n_seqlets), each motif in the middle of a 30-bp
    pattern with uniform flanks of near-zero contribution, so the trimming keeps the motif. Returns the keys."""
    import h5py

    keys, index = [], {"pos": 0, "neg": 0}
    with h5py.File(path, "w") as fh:
        fh.attrs["window_size"] = window
        for posneg, ppm, n in items:
            g = fh.require_group(f"{posneg}_patterns").create_group(f"pattern_{index[posneg]}")
            keys.append(f"{posneg}.pattern_{index[posneg]}")
            index[posneg] += 1
            a = (30 - len(ppm)) // 2
            seq = np.full((30, 4), 0.25)
            seq[a : a + len(ppm)] = ppm
            cwm = seq * 0.002
            cwm[a : a + len(ppm)] = ppm * (1.0 if posneg == "pos" else -1.0)
            g.create_dataset("sequence", data=seq)
            g.create_dataset("contrib_scores", data=cwm)
            g.create_dataset("hypothetical_contribs", data=cwm)
            s = g.create_group("seqlets")
            s.create_dataset("n_seqlets", data=np.array([n]))
            s.create_dataset("example_idx", data=np.arange(n) % 50)
            s.create_dataset("start", data=np.full(n, 185))
            s.create_dataset("end", data=np.full(n, 215))
            s.create_dataset("is_revcomp", data=np.zeros(n, bool))
    return keys


SELFTEST_MODELS = {
    # (name, head): (kind, role, [(posneg, motif, n_seqlets)] or None (no h5), (counts r, median JSD) or None)
    ("cbp_corrected", "profile"): ("chrombpnet", "corrected", [("pos", "TF0", 300), ("pos", "TF1", 200),
                                   ("pos", "TF2", 100), ("pos", "TF3", 50), ("pos", "DNASE_1", 50)], None),
    ("cbp_uncorrected", "profile"): ("chrombpnet", "uncorrected", [("pos", "TF0", 300), ("pos", "TF1", 150),
                                     ("pos", "TN5_1", 400), ("pos", "BIASX", 150)], None),
    ("current", "profile"): ("cherimoya", "baseline", [("pos", "TF0", 300), ("pos", "TF1", 200), ("pos", "TF2", 100),
                             ("pos", "TF3", 80), ("pos", "TN5_1", 120), ("neg", "BIASX", 100)], (0.764252, 0.232805)),
    ("tn5log", "profile"): ("cherimoya", "arm", [("pos", "TF0", 320), ("pos", "TF1", 210), ("pos", "TF2", 90),
                            ("pos", "DNASE_1", 40)], (0.7600, 0.2335)),
    ("tn5lin", "profile"): ("cherimoya", "arm", [("pos", "TF0", 300), ("pos", "TF1", 200), ("pos", "TF2", 100),
                            ("pos", "BIASX", 200)], (0.7700, 0.2400)),
    ("tn5many", "profile"): ("cherimoya", "arm", [("pos", "TF0", 300), ("pos", "TF1", 200), ("pos", "TF2", 100),
                             ("pos", "TN5_1", 10), ("pos", "DNASE_1", 10), ("neg", "BIASX", 10)], (0.7643, 0.2328)),
    ("tn5t2", "profile"): ("cherimoya", "arm", [("pos", "TF0", 300), ("pos", "TF4", 100)], (0.7500, 0.2300)),
    ("headft", "profile"): ("cherimoya", "arm", [("pos", "TF0", 320), ("pos", "TF1", 210), ("pos", "TF2", 90),
                            ("pos", "TF3", 60)], (0.7650, 0.2320)),
    ("tn5noperf", "profile"): ("cherimoya", "arm", [("pos", "TF0", 320), ("pos", "TF1", 210), ("pos", "TF2", 90)],
                               None),
    ("tn5miss", "profile"): ("cherimoya", "arm", None, None),
    ("cbp_corrected_tm", "profile"): ("chrombpnet", "reference", [("pos", "TF0", 300), ("pos", "TF1", 200),
                                      ("pos", "TF2", 100), ("pos", "TF3", 50)], None),
    ("cbp_corrected", "counts"): ("chrombpnet", "corrected", [("pos", "TF0", 200), ("pos", "TF1", 100),
                                  ("pos", "TF2", 50)], None),
    ("current", "counts"): ("cherimoya", "baseline", [("pos", "TF0", 200), ("pos", "TF1", 100), ("pos", "TF2", 50),
                            ("pos", "TN5_1", 30)], (0.764252, 0.232805)),
    ("tn5log", "counts"): ("cherimoya", "arm", [("pos", "TF0", 200), ("pos", "TF1", 100)], (0.7600, 0.2335)),
}  # fmt: skip
"""cbp_uncorrected's uncorrected-only patterns: TN5_1 (also a bias_db entry) and BIASX (in no database)."""
SELFTEST_VERDICTS = {"tn5log": ("PASS", []), "tn5lin": ("FAIL", ["T1", "T4"]), "tn5many": ("FAIL", ["T1"]),
                     "tn5t2": ("FAIL", ["T2", "T3"]), "headft": ("FAIL", ["guard"]),
                     "tn5noperf": ("undetermined", ["T3", "T4"]), "tn5miss": ("missing", [])}  # fmt: skip


GUARD_VARIANTS = [
    # (a finetune.json, the guard's pass): finetune_head.py's layout, then the other spellings control_norms reads
    ({"status": "ok", "checkpoint_used": "x.torch",
      "control_channel_norms": {"torch": {"fconv": 0.2, "linear": 0.1}, "final.torch": {"fconv": 0.0, "linear": 0.0}}},
     True),
    ({"status": "failed", "reason": "the frozen trunk changed",
      "control_channel_norms": {"torch": {"fconv": 0.2, "linear": 0.1}}}, False),
    ({"checkpoint": "x.final.torch", "norms": {"final": {"fconv_control": 0.1, "linear_control": 0.2}}}, True),
    ({"control_norms": {"fconv": 0.3, "linear": 0.0}}, False),
    ({"epochs": 5}, None),
]  # fmt: skip


def _selftest_world(root: Path) -> tuple[Path, dict]:
    """The databases (motif_db TF0-TF39; bias_db the same, then TN5_1 and DNASE_1), one h5 per model and head of
    SELFTEST_MODELS, the compare.tsv files (an arm's with a control column), headft's finetune.json (.torch used,
    its fconv norm 0) and spec.json. Returns the spec and {(name, head): [(key, motif)]}."""
    rng = np.random.default_rng(7)
    names = [f"TF{i}" for i in range(40)] + ["TN5_1", "DNASE_1", "BIASX"]
    mats = {n: rng.dirichlet([0.15] * 4, size=12) for n in names}
    motif_db = _write_meme(root / "motif_db.meme", {f"{n}_0": mats[n] for n in names[:40]})
    bias_db = _write_meme(root / "bias_db.meme", {(f"{n}_0" if n.startswith("TF") else n): mats[n] for n in names[:42]})
    header = ["dataset", "fold", "model", "n_regions", "counts_pearson", "counts_spearman", "counts_mse", "median_jsd",
              "median_norm_jsd"]  # fmt: skip
    entries, built = [], {}
    for (name, head), (kind, role, items, performance) in SELFTEST_MODELS.items():
        d = root / name
        d.mkdir(exist_ok=True)
        h5, tsv = d / f"{head}_modisco_results.h5", d / "compare.tsv"
        if items is not None:
            keys = _write_modisco(h5, [(pn, mats[motif], n) for pn, motif, n in items])
            built[(name, head)] = list(zip(keys, (motif for _, motif, _ in items)))
        if performance is not None:
            rows = [header + (["control"] if role == "arm" else []),
                    ["ds", "0", "chrombpnet", "100", "0.741841", "0.69", "0.52", "0.251388", "0.57"],
                    ["ds", "0", "cherimoya", "100", str(performance[0]), "0.73", "0.51", str(performance[1]), "0.61"]]  # fmt: skip
            if role == "arm":  # compare_chrombpnet.py's control column (CONTROL_COLUMN)
                rows[1].append("n/a")
                rows[2].append("real")
            tsv.write_text("".join("\t".join(r) + "\n" for r in rows))
        entries.append({"name": name, "kind": kind, "head": head, "role": role, "modisco": str(h5),
                        "performance": str(tsv) if performance is not None or name == "tn5miss" else None,
                        "finetune": str(d / "finetune.json") if name == "headft" else None})  # fmt: skip
    (root / "headft" / "finetune.json").write_text(json.dumps({
        "status": "no-op", "exit": 3, "epochs": 5, "checkpoint_used": str(root / "headft" / "cherimoya.fold_0.torch"),
        "control_channel_norms": {"torch": {"fconv": 0.0, "linear": 0.013}, "final.torch": {"fconv": 0.021,
                                                                                           "linear": 0.014}},
    }))  # fmt: skip
    spec = root / "spec.json"
    spec.write_text(json.dumps({"models": entries, "motif_db": str(motif_db), "bias_db": str(bias_db),
                                "thresholds": {"t1_share_margin": 0.02}, "dataset": "ds", "fold": 0}, indent=2))  # fmt: skip
    return spec, built


def _check_exact(root: Path, check) -> None:
    """The run with _identity_tomtom: every verdict, the numbers behind tn5log's, the counts head's."""
    import pandas as pd

    dec = json.loads((root / "decision.json").read_text())
    prof = {a["model"]: a for a in dec["heads"]["profile"]["arms"]}
    for arm, (verdict, failed) in SELFTEST_VERDICTS.items():
        got = prof.get(arm, {})
        check(got.get("verdict") == verdict, f"{arm}: verdict {got.get('verdict')}, expected {verdict}")
        reasons = [r.split(":")[0] for r in got.get("reasons", [])]
        check(reasons == failed, f"{arm}: reasons {reasons}, expected {failed}")
    t = prof["tn5log"]["tests"]
    check((t["T1"]["n_tn5like"], t["T1"]["n_limit"]) == (1, 2), f"tn5log T1 counts {t['T1']}")
    check(abs(t["T1"]["share"] - 40 / 660) < 1e-12 and abs(t["T1"]["share_limit"] - (50 / 700 + 0.02)) < 1e-12,
          f"tn5log T1 shares {t['T1']}")  # fmt: skip
    check((t["T2"]["matched"], t["T2"]["of"], t["T2"]["baseline_matched"], t["T2"]["limit"]) == (3, 4, 4, 3),
          f"tn5log T2 {t['T2']}")  # fmt: skip
    check(abs(t["T3"]["limit"] - 0.754252) < 1e-9 and abs(t["T4"]["limit"] - 0.233805) < 1e-9,
          f"tn5log T3/T4 limits {t['T3']} {t['T4']}")  # fmt: skip
    check(prof["tn5log"].get("control") == "real", f"tn5log control {prof['tn5log'].get('control')}")
    g = prof["headft"].get("guard") or {}
    check(g.get("fconv_norm") == 0.0 and g.get("linear_norm") == 0.013 and g.get("pass") is False, f"headft guard {g}")
    refs = dec["heads"]["profile"]["references"]
    check(refs["uncorrected_only"] == ["pos.pattern_2", "pos.pattern_3"], f"profile uncorrected-only {refs}")
    models = dec["models"]
    cur = models["current.profile"]
    check((cur["tn5like"], cur["tf_matched"]) == (2, 4) and abs(cur["tn5like_share"] - 220 / 900) < 1e-12,
          f"current.profile {cur}")  # fmt: skip
    check(models["cbp_corrected.profile"]["tn5like"] == 1, "cbp_corrected: DNASE_1 is its one tn5like pattern")
    check(models["cbp_uncorrected.profile"]["tn5like"] == 2, "cbp_uncorrected: TN5_1 and BIASX are tn5like")
    check(models["tn5miss.profile"]["state"] == "missing", "tn5miss is missing")
    cnt = dec["heads"]["counts"]
    check(cnt["references"]["uncorrected"] == "cbp_uncorrected.profile" and bool(cnt["references"]["note"]),
          f"the counts head takes the profile head's uncorrected-only patterns: {cnt['references']}")  # fmt: skip
    for a in cnt["arms"]:
        keys = [k for t in a["tests"].values() for k in t if k == "pass" or k.endswith("_pass")]
        check("verdict" not in a and "guard" not in a and not keys, f"counts head {a['model']} carries a verdict")
    ca = {a["model"]: a["tests"] for a in cnt["arms"]}
    check((ca["tn5log"]["T1"]["n_tn5like"], ca["tn5log"]["T2"]["matched"]) == (0, 2), f"tn5log.counts {ca['tn5log']}")
    check((models["current.counts"]["tn5like"], models["current.counts"]["tf_matched"]) == (1, 3),
          f"current.counts {models['current.counts']}")  # fmt: skip
    pats = pd.read_csv(root / "patterns.tsv", sep="\t", keep_default_na=False, na_values=["NA", ""])
    check(len(pats) == sum(len(v[2] or []) for v in SELFTEST_MODELS.values()), "one patterns.tsv row per pattern")
    lin = pats[(pats["model"] == "tn5lin") & (pats["head"] == "profile")].set_index("pattern")
    check(str(lin.loc["pos.pattern_3", "bias_flag"]) == "False" and str(lin.loc["pos.pattern_3", "tn5like"]) == "True"
          and lin.loc["pos.pattern_3", "tn5like_by"] == "uncorrected-only",
          "tn5lin's BIASX is tn5like by its uncorrected-only match alone")  # fmt: skip
    summary = pd.read_csv(root / "summary.tsv", sep="\t", keep_default_na=False, na_values=["NA", ""])
    check(len(summary) == len(SELFTEST_MODELS), "one summary.tsv row per model and head")
    md = (root / "decision.md").read_text()
    check(md.startswith("# tn5 pilot: decision, dataset ds, fold 0\n"), f"decision.md title {md.splitlines()[0]}")
    check("**PASS**" in md and "**missing**" in md and ("EDT" in md or "EST" in md), "decision.md verdicts and time")


def _check_tomtom(root: Path, built: dict, check) -> None:
    """The run with tomtom-lite: what identical copies guarantee (bias flags of TN5_1/DNASE_1 copies and of none of
    the TF copies, TF copies matched)."""
    import pandas as pd

    pats = pd.read_csv(root / "patterns.tsv", sep="\t", keep_default_na=False, na_values=["NA", ""])
    for (name, head), items in built.items():
        part = pats[(pats["model"] == name) & (pats["head"] == head)]
        flags = dict(zip(part["pattern"], part["bias_flag"].astype(str)))
        check(len(flags) == len(items), f"tomtom-lite: {name}.{head} has {len(flags)} rows, not {len(items)}")
        for key, motif in items:
            if motif in ("TN5_1", "DNASE_1"):
                check(flags.get(key) == "True", f"tomtom-lite: {name}.{head} {key} ({motif}) is not flagged")
            elif motif.startswith("TF"):
                check(flags.get(key) == "False", f"tomtom-lite: {name}.{head} {key} ({motif}) is flagged")
    dec = json.loads((root / "decision.json").read_text())
    u = dec["heads"]["profile"]["references"]["uncorrected_only"] or []
    check("pos.pattern_0" not in u and "pos.pattern_1" not in u, f"tomtom-lite: TF0/TF1 are uncorrected-only: {u}")
    m = dec["models"]
    cur, log = m["current.profile"], m["tn5log.profile"]
    check(cur["tf_matched"] == cur["corrected_tf"], f"tomtom-lite: current matches {cur['tf_matched']} TF patterns")
    check(cur["corrected_tf"] - 1 <= log["tf_matched"] <= cur["corrected_tf"],
          f"tomtom-lite: tn5log matches {log['tf_matched']} of {cur['corrected_tf']} TF patterns")  # fmt: skip


def selftest() -> int:
    """Two runs on SELFTEST_MODELS: with _identity_tomtom (every match known, so the verdicts and numbers are checked
    exactly), then through the CLI with tomtom-lite (_check_tomtom). Exit 0 iff every check holds."""
    problems: list[str] = []

    def check(ok: bool, what: str) -> None:
        if not ok:
            problems.append(what)
            logger.error("selftest: %s", what)

    with tempfile.TemporaryDirectory(prefix="compare_arms_selftest.") as tmp:
        root = Path(tmp)
        spec, built = _selftest_world(root)
        real = motifs.tomtom_p
        motifs.tomtom_p = _identity_tomtom
        try:
            run(spec, root / "exact")
        except Exception as exc:  # any failure fails the selftest, with its reason
            check(False, f"the run with the exact stand-in failed: {type(exc).__name__}: {exc}")
        finally:
            motifs.tomtom_p = real
        if not problems:
            try:
                _check_exact(root / "exact", check)
            except Exception as exc:  # a missing key or file is a failed check too
                check(False, f"checking the exact run: {type(exc).__name__}: {exc}")
        for variant, want in GUARD_VARIANTS:
            path = root / "finetune_variant.json"
            path.write_text(json.dumps(variant))
            got = guard(path)
            check(got["pass"] is want, f"guard of {variant}: {got}")
        rc = main(["--spec", str(spec), "--out-dir", str(root / "tomtom")])
        check(rc == 0, f"the CLI with tomtom-lite exited {rc}")
        if rc == 0:
            try:
                _check_tomtom(root / "tomtom", built, check)
            except Exception as exc:
                check(False, f"checking the tomtom-lite run: {type(exc).__name__}: {exc}")
    print(f"compare_arms selftest: {f'{len(problems)} check(s) failed' if problems else 'all checks passed'}",
          flush=True)  # fmt: skip
    return 1 if problems else 0


# ---------------------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--spec", type=Path, help="spec.json (see above)")
    what.add_argument("--selftest", action="store_true", help="the synthetic check; exit 0 iff it passes")
    ap.add_argument("--out-dir", type=Path, help="where patterns.tsv, summary.tsv and decision.json/.md go")
    ap.add_argument("--threads", type=int, help="tomtom-lite's threads (NUMBA_NUM_THREADS)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    common.setup_logging(args.verbose)
    if args.threads:
        os.environ["NUMBA_NUM_THREADS"] = str(args.threads)
    if args.selftest:
        return selftest()
    if args.out_dir is None:
        ap.error("--spec needs --out-dir")
    try:
        note = run(args.spec, args.out_dir)
    except (OSError, ValueError, KeyError) as exc:
        logger.error("%s", exc)
        common.result(f"failed: {exc}")
        return 1
    common.result(note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
