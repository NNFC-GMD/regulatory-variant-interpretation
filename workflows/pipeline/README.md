# workflows/pipeline — one project, end to end, in one box

A project is a directory and a `project.yaml` in it. From its fragment files to Cherimoya models, attributions,
TF-MoDISco motifs, one compendium, Fi-NeMo hits and DeepISA cooperativity, every step runs from here.
[docs/pipeline.md](../../docs/pipeline.md) explains each step and why it is set up as it is;
[docs/new_project.md](../../docs/new_project.md) walks through starting a project.

| File | What it does |
|---|---|
| `project.example.yaml` | every key of a project.yaml, with its default |
| `project.py` | loads and checks project.yaml, derives every path, says which stages are done (`status`, `check <stage>`), writes DeepISA's inputs |
| `fetch_igvf.py` | the IGVF portal files a project names, md5-checked, into `<root>/fragments/<stem>.fragments.tsv.gz` (login node) |
| `submit.sh` | the checks that need no box (clean, pushed checkout; igvf at `igvf.pin`; inputs; dry run; no other box), then `sbatch run_project.sbatch` |
| `run_project.sbatch` | one whole DCAI node for `run_project.sh` |
| `run_project.sh` | the stages in order: preflight, prep, train, attribute, downstream, motifs, cooperativity, coop_matrix (only when project.yaml lists it), summary; `--dry-run` prints them |
| `status.py` | `<root>/STATUS.md`: each stage's state and each dataset's headline numbers |
| `prune.py` | `project.py <yaml> prune`: a finished project's derived intermediates, by rule, into a manifest a person reads; `--apply <manifest>` deletes exactly those files. Never what a stage check reads (docs/results.md) |
| `share.py` | `project.py <yaml> share --out <file.tar>`: the results in one tar, with an md5 of each file, for someone without cluster access |
| `lib.sh` | the shell helpers (log in ET, stage records, CPU blocks, the hold after a failure) |
| `igvf.pin` | the igvf_tf_chrombpnet commit the steps come from |

```bash
pixi run -e default python workflows/pipeline/fetch_igvf.py <root>/project.yaml      # login node
bash workflows/pipeline/run_project.sh <root>/project.yaml --dry-run                 # login node
bash workflows/pipeline/submit.sh <root>/project.yaml                                # the box
pixi run -e default python workflows/pipeline/project.py <root>/project.yaml status  # any time
pixi run -e default python workflows/pipeline/project.py <root>/project.yaml prune   # when finished: the manifest
```

A project set up by hand before the driver (kept for its results) says `driver: false` in its project.yaml: `check`,
`status`, `prune` and `share` work on it, and the driver (`env`, `coop-inputs`, so `run_project.sh` and `submit.sh`)
refuses it.
