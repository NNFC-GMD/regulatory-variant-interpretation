"""Tn5 shift detection, so preprocessing needs no chrombpnet.

Vendored from scPrinter (see attribution below) and moved here from
scripts/python/ because it is pipeline code now, not a standalone tool.

Two deliberate changes from upstream:

1. **torch is gone.** ``DNA_one_hot`` built a torch tensor that the only caller
   immediately turned back into numpy, so it cost a multi-GB dependency in the
   preprocessing environment for nothing.
2. **pyfaidx and the genome FASTA are imported lazily**, inside the detection
   functions. Detection needs sequence around the cut sites by its nature, but
   the pileup does not: give ``plus_shift``/``minus_shift`` in the config and
   neither a FASTA nor pyfaidx is touched at all.
3. **What the detector measures, and what it returns.** ``circular_detect``
   finds the offset ``d`` at which the sample's cut-site base composition best
   matches the reference matrices below, and those matrices are in scPrinter's
   own frame: fragments already shifted +4/-5, as 10x/CellRanger write them.
   So the shift PRESENT in the reads is ``(4 - d_plus, -5 - d_minus)`` --
   exactly what upstream ``detect_shift`` returns -- and ``d`` itself is not a
   shift at all. :func:`detect_shift_raw` returns the shift present (same
   numbers as ``detect_shift``, under the name the callers use), and
   :func:`shift_deltas` turns it into chrombpnet's +4/-4 (ATAC) or 0/+1
   (DNASE) correction.

   An earlier version of this module read ``detect_shift``'s return value as
   deltas and returned the bare ``d`` as "the shift present". On 10x
   fragments that reported (0, 0) -- with a near-perfect fit -- where
   chrombpnet's own detector (``auto_shift_detect.compute_shift``, whose
   reference is unshifted BAM reads) reports +4/-5 for the same file, and
   ``shift_deltas`` then applied +4/-4 instead of 0/+1: every insertion four
   bases off. Checked on a 10x multiome fragments file (cellranger-arc 2.0.2),
   525,245 fragments sampled genome-wide: scPrinter ``d`` = (0, 0) at MSE
   1e-4; chrombpnet +4/-5 at two seeds.
"""

# Author: Ruochi Zhang
# Affilitation: Broad Institute of MIT and Harvard
# Original code: https://github.com/buenrostrolab/scPrinter/blob/main/scprinter/shift_detection.py

import gzip
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DNAmapping = np.zeros((128, 4), dtype=int)
DNAmapping[65] = [1, 0, 0, 0]  # A
DNAmapping[67] = [0, 1, 0, 0]  # C
DNAmapping[71] = [0, 0, 1, 0]  # G
DNAmapping[84] = [0, 0, 0, 1]  # T
DNAmapping[78] = [0, 0, 0, 0]  # N


# Hardcoded reference bias matrices
ref_forward_bias, ref_reverse_bias = (
    np.array(
        [
            [0.22801, 0.267627, 0.270118, 0.234245],
            [0.230582, 0.263092, 0.266591, 0.239735],
            [0.230966, 0.263395, 0.277853, 0.227786],
            [0.237373, 0.254811, 0.268621, 0.239195],
            [0.227756, 0.262171, 0.281433, 0.22864],
            [0.229343, 0.267518, 0.266358, 0.236781],
            [0.233688, 0.261873, 0.268756, 0.235683],
            [0.233246, 0.258417, 0.267473, 0.240864],
            [0.233571, 0.250868, 0.278438, 0.237123],
            [0.210302, 0.265267, 0.273532, 0.250899],
            [0.197471, 0.270346, 0.276262, 0.255921],
            [0.239908, 0.224086, 0.353718, 0.182288],
            [0.209381, 0.325079, 0.290731, 0.174809],
            [0.27844, 0.226113, 0.273372, 0.222075],
            [0.174853, 0.40936, 0.237924, 0.177863],
            [0.345859, 0.149262, 0.228952, 0.275927],
            [0.165938, 0.270514, 0.480596, 0.082952],
            [0.180195, 0.29281, 0.221314, 0.305681],
            [0.179562, 0.332162, 0.237392, 0.250884],
            [0.125249, 0.409938, 0.152056, 0.312757],
            [0.334499, 0.169781, 0.172222, 0.323498],
            [0.300765, 0.151369, 0.411777, 0.136089],
            [0.255409, 0.225268, 0.345623, 0.1737],
            [0.294189, 0.221911, 0.303888, 0.180012],
            [0.125051, 0.422991, 0.270467, 0.181491],
            [0.25149, 0.249151, 0.18284, 0.316519],
            [0.190253, 0.229807, 0.393018, 0.186922],
            [0.222202, 0.265248, 0.242275, 0.270275],
            [0.171303, 0.27846, 0.331291, 0.218946],
            [0.182575, 0.338698, 0.238956, 0.239771],
            [0.249742, 0.270574, 0.275662, 0.204022],
            [0.242147, 0.26794, 0.272813, 0.2171],
            [0.238624, 0.26772, 0.261579, 0.232077],
            [0.234333, 0.259208, 0.264034, 0.242425],
            [0.234719, 0.257802, 0.269948, 0.237531],
            [0.232542, 0.259099, 0.274311, 0.234048],
            [0.228187, 0.266636, 0.273797, 0.23138],
            [0.231669, 0.263802, 0.26339, 0.241139],
            [0.228852, 0.26637, 0.274964, 0.229814],
            [0.23085, 0.260809, 0.266385, 0.241956],
        ]
    ),
    np.array(
        [
            [0.227808, 0.264997, 0.272057, 0.235138],
            [0.229786, 0.261645, 0.267999, 0.24057],
            [0.229542, 0.265141, 0.277305, 0.228012],
            [0.2364, 0.25577, 0.267527, 0.240303],
            [0.228397, 0.261634, 0.278268, 0.231701],
            [0.227913, 0.266164, 0.267063, 0.23886],
            [0.233518, 0.263004, 0.268147, 0.235331],
            [0.231852, 0.259135, 0.267172, 0.241841],
            [0.233683, 0.251586, 0.27724, 0.237491],
            [0.209089, 0.266759, 0.272593, 0.251559],
            [0.200028, 0.266882, 0.283073, 0.250017],
            [0.235364, 0.228948, 0.348828, 0.18686],
            [0.213109, 0.320979, 0.290091, 0.175821],
            [0.271552, 0.238546, 0.267749, 0.222153],
            [0.186768, 0.377292, 0.242154, 0.193786],
            [0.312909, 0.176144, 0.256335, 0.254612],
            [0.176285, 0.258176, 0.440994, 0.124545],
            [0.174572, 0.297044, 0.226605, 0.301779],
            [0.174146, 0.338917, 0.232244, 0.254693],
            [0.129944, 0.406257, 0.1538, 0.309999],
            [0.329279, 0.166597, 0.17534, 0.328784],
            [0.301616, 0.149981, 0.416916, 0.131487],
            [0.251985, 0.230058, 0.337205, 0.180752],
            [0.300088, 0.215177, 0.299529, 0.185206],
            [0.083786, 0.462517, 0.282588, 0.171109],
            [0.274189, 0.22066, 0.155771, 0.34938],
            [0.174265, 0.225577, 0.425875, 0.174283],
            [0.219996, 0.271626, 0.230628, 0.27775],
            [0.171276, 0.28058, 0.333658, 0.214486],
            [0.178319, 0.342787, 0.233408, 0.245486],
            [0.255848, 0.265675, 0.27829, 0.200187],
            [0.241528, 0.267603, 0.273153, 0.217716],
            [0.238694, 0.266755, 0.261775, 0.232776],
            [0.234914, 0.2593, 0.26413, 0.241656],
            [0.234639, 0.258018, 0.270217, 0.237126],
            [0.231615, 0.258526, 0.275489, 0.23437],
            [0.226413, 0.269257, 0.273205, 0.231125],
            [0.230697, 0.264718, 0.261399, 0.243186],
            [0.228284, 0.265556, 0.273413, 0.232747],
            [0.230259, 0.260566, 0.268176, 0.240999],
        ]
    ),
)


def DNA_one_hot(sequence, dtype="float", device="cpu"):
    """
    Convert a DNA sequence string into a one-hot encoding tensor.

    Parameters
    ----------
    sequence : str
        DNA sequence string.
    dtype : str
        Data type of the returned tensor.
    device : str
        Device of the returned tensor.
    verbose : bool
        If True, print progress.

    Returns
    -------
    np.ndarray
        One-hot encoding tensor.
    """
    if type(sequence) is list or type(sequence) is np.ndarray:
        sequence = "".join(sequence)
    ascii_values = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    # Use the ASCII values to index into the mapping array
    one_hot_encoded = DNAmapping[ascii_values]
    # Upstream returned a torch int8 tensor here and the caller called
    # on it straight away; numpy directly avoids the torch dependency.
    return one_hot_encoded.astype(np.int8).T


def get_nucleotide_freq(beds, genome, context_radius=20, paired=True):
    from pyfaidx import Fasta  # noqa: PLC0415  # imported only when detecting

    """
    Get the nucleotide frequency around the start and end of the fragments in the bed file this is for paired end
    Parameters
    ----------
    beds
    genome
    context_radius

    Returns
    -------

    """
    # `genome` is a FASTA PATH here, not scPrinter's genome object. Upstream
    # called genome.fetch_fa() and genome.chrom_sizes; this pipeline has no such
    # object, only the file, so both come from the path and its pyfaidx index.
    genome_seq = Fasta(str(genome))
    known_chroms = set(genome_seq.keys())
    forward_bias = []
    reverse_bias = []
    skipped = 0
    for frags in np.array(beds):
        chrom, start, end = frags[0], int(frags[1]), int(frags[2])
        if chrom not in known_chroms:
            # print(f"Warning: Chromosome {chrom} not found in the genome file")
            continue
        if not paired:
            strand = [frags[5]]
        else:
            strand = ["+", "-"]
        # A cut site within context_radius of a contig end yields a SHORT
        # sequence: pyfaidx clamps the slice rather than padding it. Upstream
        # noticed the wrong shape, printed it, and appended the row anyway,
        # which makes the stack below inhomogeneous and np.mean raise
        # "setting an array element with a sequence". A truncated context has
        # no meaningful per-position frequency, so drop it and count it.
        if "+" in strand:
            forward_context = DNA_one_hot(
                genome_seq[chrom][start - context_radius : start + context_radius].seq.upper()
            ).T
            if forward_context.shape[0] != 2 * context_radius:
                skipped += 1
            else:
                forward_bias.append(forward_context)
        if "-" in strand:
            reverse_context = DNA_one_hot(
                genome_seq[chrom][end - context_radius : end + context_radius].seq.upper()
            ).T
            if reverse_context.shape[0] != 2 * context_radius:
                skipped += 1
            else:
                reverse_bias.append(reverse_context)

    if skipped:
        logger.info(f"shift detection: skipped {skipped} context(s) truncated by a contig edge")
    if not forward_bias or not reverse_bias:
        raise ValueError(
            "shift detection: no usable cut-site contexts "
            f"(forward={len(forward_bias)}, reverse={len(reverse_bias)}). "
            "Every sampled fragment sat within "
            f"{context_radius}bp of a contig end, or on a contig absent from the genome. "
            "Set plus_shift/minus_shift in the dataset config to skip detection."
        )

    forward_bias = np.mean(np.array(forward_bias), axis=0)
    reverse_bias = np.mean(np.array(reverse_bias), axis=0)

    return forward_bias, reverse_bias


def circular_shift(matrix, shift):
    """
    Circular shift a matrix by a given distance
    :param matrix: Matrix to be shifted
    :param shift: Shift distance
    :return: Shifted matrix
    """
    return np.concatenate((matrix[-shift:, :], matrix[:-shift, :]), axis=0)


def mse(matrix1, matrix2):
    """
    Compute mean squared error between two matrices
    :param matrix1: Input matrix 1
    :param matrix2: Input matrix 2
    :return: MSE valu
    """
    return np.mean((matrix1 - matrix2) ** 2)


def circular_detect(ref_bias, query_bias):
    """
    Detect shift distance between two bias matrices
    :param ref_bias: Reference bias matrix
    :param query_bias: Query bias matrix
    :return: Shift distance
    """

    # Make sure two matrices are the same shape
    assert np.shape(ref_bias) == np.shape(query_bias), "Input matrices must have the same shape"

    # Circular shift the query bias matrix and compute MSE at each shift distance
    context_radius = int(np.shape(ref_bias)[0] / 2)
    shift_mse = [
        mse(circular_shift(ref_bias, shift), query_bias)
        for shift in range(-context_radius, context_radius)
    ]
    minimum_mse = np.min(shift_mse)
    if minimum_mse > 0.002:
        print(f"Warning: The shift detection may not be accurate. The minimum MSE is {minimum_mse}")
    else:
        print(f"Minimum MSE is {minimum_mse}, shift detected")
    # print (np.min(shift_mse), shift_mse)
    # Return the shift distance that minimizes MSE
    return np.argmin(shift_mse) - context_radius


def count_lines(filename):
    clip = 1000000

    if ".gz" in filename:
        file = gzip.open(filename, "rt")
    else:
        file = open(filename)
    ct = 0
    for _line in file:
        ct += 1
        if ct > clip:
            break
    file.close()
    return ct


def sample_bed_file_to_dataframe(filename, sample_size):
    """
    randomly sample a fragment file and convert it to a pandas DataFrame
    """
    import random

    line_count = count_lines(filename)
    # add 100 to the sample size to account for the header / comment lines etc.
    sample_indices = set(random.sample(range(line_count), sample_size + 100))

    samples = []
    if ".gz" in filename:
        file = gzip.open(filename, "rt")
    else:
        file = open(filename)
    for i, line in enumerate(file):
        if i in sample_indices:
            if line.startswith("#"):
                continue
            samples.append(line.strip().split("\t"))
            if len(samples) == sample_size:
                break

    file.close()
    # Convert to DataFrame
    df = pd.DataFrame(samples)
    return df


def detect_shift(frags, genome):
    """
    Detect the +/- strand shifting that has been done on the paired end or single end fragments file.

    Parameters
    ----------
    frags : str
        Path to the fragments file.
    genome : str
        Path to the reference genome file.

    Returns
    -------
    tuple
        A tuple containing the detected shift distance for the forward strand and the reverse strand.
        The shift distance is represented as the number of positions shifted.

    """

    # If the fragments file is too large, sample a smaller subset
    if os.path.getsize(frags) / (1024**3) < 2:
        frags = pd.read_csv(frags, sep="\t", header=None, comment="#").sample(10000)
    else:
        frags = sample_bed_file_to_dataframe(frags, 10000)

    single_end = False
    if len(frags.columns) >= 6:
        if str(frags.iloc[0, 5]) in ["+", "-"]:
            single_end = True
    print(
        "Detecting the shift in the", "single end" if single_end else "paired end", "fragments file"
    )
    print("If you think the above message is wrong, please check the input file format")
    forward_bias, reverse_bias = get_nucleotide_freq(frags, genome, paired=not single_end)
    return 4 - circular_detect(ref_forward_bias, forward_bias), -5 - circular_detect(
        ref_reverse_bias, reverse_bias
    )


def shift_present(detected_plus: int, detected_minus: int) -> tuple[int, int]:
    """The shift present in the reads, from ``circular_detect``'s offsets.

    The reference bias matrices are in the +4/-5 frame, so an offset of 0
    means +4/-5 already applied (10x fragments) and an offset of (+4, -5)
    means raw, unshifted reads -- see the module docstring.
    """
    return 4 - int(detected_plus), -5 - int(detected_minus)


def detect_shift_raw(frags, genome):
    """The Tn5 shift actually present in the reads, as (plus, minus).

    The same numbers as upstream ``detect_shift``: the reference matrices sit
    in the +4/-5 frame, so the shift present is ``(4 - d, -5 - d)`` for the
    detected offsets ``d`` (``shift_present``). Feed it to ``shift_deltas``
    for chrombpnet's correction.

    For a file over 2 GB this samples only the first million lines, which in
    a coordinate-sorted fragments file is the start of chr1; there the fit can
    fail (scPrinter warns above MSE 0.002). Sample genome-wide first if so --
    workflows/dcai/detect_shifts.py does.
    """
    if os.path.getsize(frags) / (1024**3) < 2:
        df = pd.read_csv(frags, sep="\t", header=None, comment="#").sample(10000)
    else:
        df = sample_bed_file_to_dataframe(frags, 10000)

    single_end = False
    if len(df.columns) >= 6 and str(df.iloc[0, 5]) in ["+", "-"]:
        single_end = True

    forward_bias, reverse_bias = get_nucleotide_freq(df, genome, paired=not single_end)
    return shift_present(
        circular_detect(ref_forward_bias, forward_bias),
        circular_detect(ref_reverse_bias, reverse_bias),
    )


def shift_deltas(detected_plus: int, detected_minus: int, assay: str = "ATAC"):
    """Deltas that move reads carrying a given shift onto chrombpnet's convention.

    ``detected_plus/minus`` is the shift PRESENT in the reads (``detect_shift_raw``,
    or the config's plus_shift/minus_shift). Matches chrombpnet 1.0.1
    reads_to_bigwig.main exactly::

        ATAC : (4 - plus, -4 - minus)
        DNASE: (-plus,     1 - minus)

    The target is +4/-4, not the +4/-5 most fragment files carry: 10x fragments
    (+4/-5 present) get (0, +1).
    """
    if assay == "ATAC":
        return 4 - detected_plus, -4 - detected_minus
    if assay == "DNASE":
        return -detected_plus, 1 - detected_minus
    raise ValueError(f"assay must be ATAC or DNASE, got {assay!r}")
