# A Tn5 bias track as a Cherimoya control

Cherimoya has no bias model: it learns the observed ATAC signal, Tn5's
sequence preference included. These tools test the alternative of giving it a
precomputed per-base Tn5 bias track as a control (`fit.json` `controls`), as
ChromBPNet's bias model gives its corrected model one. The track is scPrinter's
(v2): an h5 with one float32 dataset per contig, made by a small CNN that maps
L bases of one-hot sequence to L - 100 outputs (each sees +-50 bp).

The h5 holds `v = 10^((x - 0.5) * 2) - 0.01` of the CNN's output `x`, so
`log10(v + 0.01) = 2x - 1`.

| tool | environment | what it does |
|---|---|---|
| `tn5_align.py` | the cherimoya env, CPU | the alignment gate: does base i of the track describe the cut at base i, on each strand? writes `align.json` and `gate.json` |
| `tn5_tracks.py` | an env with figwig, h5py, pyBigWig | the h5 as two bigWig controls, shifted by the gate's offset, read back and checked; a JSON sidecar |
| `tn5_cnn_check.py` | the cherimoya env, GPU if there is one | the CNN against its own h5: output length, registration, strand symmetry (informational) |
| `finetune_head.py` | the cherimoya env, GPU | a trained model's heads taught the control with the trunk frozen |
| `compare_arms.py` | the cherimoya env, CPU | the per-arm decision from TF-MoDISco patterns and test performance |

Each tool's docstring (`--help`) is its full description. Every one has
`--selftest`: a tiny synthetic run in a temporary directory that reads no
real data and exits 0 only if every check passes. Outputs are written under a
`.partial` name and renamed, and each tool prints one summary line at the end.

## The alignment gate (`tn5_align.py`)

The fragments are streamed once and cut where the pipeline cuts them
(igvf_tf_chrombpnet `lib/python/utils/pileup.py`): `start + plus_delta` on the
plus strand, `end + minus_delta - 1` on the minus strand; 10x fragments carry
+4/-5, so the deltas to the pipeline's +4/-4 are (0, +1). Over windows sampled
from the negatives (background, where cuts mostly follow Tn5's own
preference), with `b = log10(v + 0.01)` and `c_s` the cuts of strand
`s` = plus, minus or both:

    m_s(d) = sum_i c_s[i] b[i + d] / sum_i c_s[i] - mean_i b[i],   d in [-5, 5]

The gate is **clean** iff `argmax_d m` is the same `d*` on plus, minus and
both, `|d*| <= 1`, `m_both(d*) > 0`, and plus + minus reproduce the training
signal bigWig at >= 0.999 of the window bases (so these are the cuts the
model is trained on). The offset is then `d*`. Otherwise the gate is
**ambiguous**, with the failed clauses as the reason: most likely a strand
split, where one strand peaks at 0 and the other at +-1, which no single
shift of the track can fix.

The fragments are decompressed by htslib's `bgzip -dc -@<threads>` when one
is found (`--bgzip`, else on PATH, else beside the interpreter); without one
the fallback is `gzip -dc` on a single thread, several times slower on a
multi-GB file, so pass `--bgzip` when the environment that runs the tool has
no htslib. `align.json` records the command used.

## The tracks (`tn5_tracks.py`)

Track base i holds `f(h5[i + offset])`, offset = the gate's when it is clean
and 0 when it is ambiguous; bases shifted off a chromosome take `f` of the N
constant (the CNN's output on N, read from the chromosome ends of the h5).

| track | f(v) | range |
|---|---|---|
| `<prefix>.log.bw` | `log10(v + 0.01) - ymin`, ymin the genome minimum | >= 0; the log scale, for a profile head that adds the control in logit space |
| `<prefix>.linear.bw` | `v + 0.01` = `10^(2x - 1)` | > 0 |

Both are computed in float64, stored as float32, every base (no zoom levels),
in the order of the chromosome sizes given. Random windows and every
chromosome's ends are read back with pyBigWig and must equal the same
transform of the h5. The sidecar `<prefix>.tracks.json` records ymin,
quantiles, the N constant, the offset and the gate it came from. A rerun with
the same offset and the same h5 does nothing.

## The CNN check (`tn5_cnn_check.py`)

On 1,100-bp windows of A, C, G and T only: the output has L - 100 bases; `2x - 1`
equals `log10(h5 + 0.01)` of the central bases to 1e-4 (offsets -3..+3
searched, so that a misregistration shows as its size); and the CNN is
strand-symmetric: `m(x)` against `m(rc(x)).flip(-1)` has Pearson >= 0.98 and
median |difference| <= 0.05 at offset 0 (offsets -2..+2 reported). Not a
gate. The CNN runs in float64, because its traced convolutions carry
`allow_tf32=True` from the time they were traced. The h5 itself may have
been made at a lower precision, so a registration fail with the best offset
at 0 and a small median difference is precision, not a slip; a slip shows as
a best offset other than 0.
