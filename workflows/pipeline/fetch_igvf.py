#!/usr/bin/env python3
"""Fetch a project's input files from the IGVF data portal, checked, into the names the pipeline reads.

For every sample of project.yaml whose fragments is an IGVF file accession (IGVFFI....), and for its index and extra
accessions, the portal's metadata (https://api.data.igvf.org/<accession>/) gives the md5sum, the size and the
download link; the file is streamed to a temporary name, its md5 compared with the portal's (and with the sample's
own md5, when project.yaml gives one), and only then renamed into place:

  fragments  <root>/fragments/<stem>.fragments.tsv.gz, a real file, not a link: igvf_tf_chrombpnet's make_configs.py
             resolves links, and its config check accepts .fragments.tsv.gz but not the portal's .bed.gz. Its first
             bytes must be a BGZF header (gzip with the BC extra field), which the peak caller and figwig need.
  index      <root>/fragments/<stem>.fragments.tsv.gz.tbi
  extra      <root>/portal/<accession>.<the portal's extension>
  every file a row in <root>/portal/manifest.tsv: accession, stem, role, path, md5, bytes, portal summary, fetched (ET)

A file already in place with the right md5 is kept. A sample whose fragments is a path is checked (BGZF, md5 when
given) and linked into fragments/ only if its name differs. Only the IGVF portal is contacted. The files are public
(controlled_access false); a controlled-access file is refused, since it needs credentials this script does not
handle.

    fetch_igvf.py <project.yaml> [--dry-run]

Standard library and PyYAML only (the default pixi env); I/O-bound, so it runs on the login node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project as pj  # noqa: E402

PORTAL = "https://api.data.igvf.org"
BGZF_MAGIC = b"\x1f\x8b\x08\x04"
CHUNK = 8 << 20


def et() -> str:
    os.environ.setdefault("TZ", "America/New_York")
    time.tzset()
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def metadata(accession: str) -> dict:
    req = urllib.request.Request(f"{PORTAL}/{accession}/?format=json", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    for key in ("md5sum", "href", "file_size"):
        if not d.get(key):
            raise RuntimeError(f"{accession}: the portal gives no {key}")
    if d.get("controlled_access"):
        raise RuntimeError(f"{accession}: controlled access; fetch it with credentials, then give its path")
    return d


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def is_bgzf(path: Path) -> bool:
    with open(path, "rb") as fh:
        head = fh.read(18)
    # gzip with FEXTRA, and the BGZF subfield 'BC' in the extra field
    return head[:4] == BGZF_MAGIC and head[12:14] == b"BC"


def download(meta: dict, dest: Path, want_md5: str) -> None:
    tmp = dest.with_name(dest.name + ".partial")
    url = PORTAL + meta["href"]
    h = hashlib.md5()
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
        while chunk := r.read(CHUNK):
            fh.write(chunk)
            h.update(chunk)
    got = h.hexdigest()
    if got != want_md5:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{meta['accession']}: md5 {got} differs from the portal's {want_md5}")
    if tmp.stat().st_size != int(meta["file_size"]):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{meta['accession']}: {tmp.stat().st_size} bytes, the portal says {meta['file_size']}")
    os.replace(tmp, dest)


def fetch_one(accession: str, dest: Path, own_md5: str | None, dry: bool) -> dict:
    meta = metadata(accession)
    want = meta["md5sum"]
    if own_md5 and own_md5 != want:
        raise RuntimeError(f"{accession}: project.yaml's md5 {own_md5} differs from the portal's {want}")
    if dest.is_file() and dest.stat().st_size == int(meta["file_size"]) and md5_of(dest) == want:
        state = "kept"
    elif dry:
        state = "would fetch"
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        download(meta, dest, want)
        state = f"fetched in {time.time() - t0:.0f} s"
    return {"accession": accession, "path": str(dest), "md5": want, "bytes": int(meta["file_size"]),
            "summary": meta.get("summary", ""), "state": state}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="check the portal and the files in place; fetch nothing")
    args = ap.parse_args(argv)
    p = pj.load(args.project)
    q = pj.paths(p)
    rows, failed = [], 0
    for s in p["samples"]:
        frag = pj.fragments_file(p, s)
        jobs = []
        if s["accession"]:
            jobs.append(("fragments", s["accession"], frag, s["md5"]))
        if s["index"]:
            jobs.append(("index", s["index"], frag.with_name(frag.name + ".tbi"), None))
        for acc in s["extra"]:
            jobs.append(("extra", acc, None, None))
        for role, acc, dest, own in jobs:
            try:
                if dest is None:  # extra: named by the portal's own file name
                    ext = metadata(acc)["href"].rsplit("/", 1)[-1].split(".", 1)[-1]
                    dest = q["portal"] / f"{acc}.{ext}"
                row = fetch_one(acc, dest, own, args.dry_run)
                if role == "fragments" and not args.dry_run and not is_bgzf(dest):
                    raise RuntimeError(f"{dest} is not BGZF-compressed (bgzip it, or the steps cannot read it)")
            except Exception as exc:  # noqa: BLE001  one file's failure is reported, the others go on
                print(f"ERROR: {s['stem']} {role} {acc}: {exc}", file=sys.stderr)
                failed += 1
                continue
            rows.append({"stem": s["stem"], "role": role, **row})
            print(f"{s['stem']}\t{role}\t{acc}\t{row['state']}\t{row['path']}")
        if not s["accession"]:  # a local path
            src = Path(s["fragments"])
            if not src.is_file() or not is_bgzf(src):
                print(f"ERROR: {s['stem']}: {src} is missing or not BGZF", file=sys.stderr)
                failed += 1
                continue
            if s["md5"] and md5_of(src) != s["md5"]:
                print(f"ERROR: {s['stem']}: {src} md5 differs from project.yaml's", file=sys.stderr)
                failed += 1
                continue
            if src.resolve() != frag.resolve() and not args.dry_run:
                frag.parent.mkdir(parents=True, exist_ok=True)
                if frag.exists() or frag.is_symlink():
                    frag.unlink()
                if src.name.endswith(pj.FRAGMENTS_SUFFIX):
                    frag.symlink_to(src)
                else:
                    raise SystemExit(f"ERROR: {src}: name it <stem>{pj.FRAGMENTS_SUFFIX} or give an accession: "
                                     "the config check rejects other suffixes, and links are resolved")  # fmt: skip
            rows.append({"stem": s["stem"], "role": "fragments", "accession": "", "path": str(frag),
                         "md5": s["md5"] or "", "bytes": src.stat().st_size, "summary": str(src),
                         "state": "linked"})  # fmt: skip
    if rows and not args.dry_run:
        q["portal"].mkdir(parents=True, exist_ok=True)
        manifest = q["portal"] / "manifest.tsv"
        cols = ["accession", "stem", "role", "path", "md5", "bytes", "summary", "fetched"]
        now = et()
        lines = ["\t".join(cols)] + ["\t".join(str(r.get(c, now if c == "fetched" else "")) for c in cols)
                                     for r in rows]  # fmt: skip
        manifest.write_text("\n".join(lines) + "\n")
        print(f"manifest: {manifest}")
    if failed:
        print(f"{failed} file(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
