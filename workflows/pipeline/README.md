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
| `lib.sh` | the shell helpers (log in ET, stage records, CPU blocks, the hold after a failure) |
| `igvf.pin` | the igvf_tf_chrombpnet commit the steps come from |

```bash
pixi run -e default python workflows/pipeline/fetch_igvf.py <root>/project.yaml      # login node
bash workflows/pipeline/run_project.sh <root>/project.yaml --dry-run                 # login node
bash workflows/pipeline/submit.sh <root>/project.yaml                                # the box
pixi run -e default python workflows/pipeline/project.py <root>/project.yaml status  # any time
```
