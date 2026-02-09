"""
Split a large group of proteins into smaller dscript prediction tasks using a blocked approach. 
"""

from __future__ import annotations

import argparse
from math import ceil
import sys
from collections.abc import Callable
from typing import NamedTuple
import os
import numpy as np

from ..utils import log


class SplitTasksArguments(NamedTuple):
    cmd: str
    protins: str | None
    pairs: str | None
    model: str | None
    embeddings: str
    foldseek_fasta: str | None
    workdir: str | None
    device: str | None
    thresh: float | None
    load_proc: int | None
    blocks: int | None
    split_blocks: int | None
    func: Callable[[SplitTasksArguments], None]


def add_args(parser):
    """
    Create parser for command line utility

    :meta private:
    """
    group_a = parser.add_argument_group('Arguments for Spliting Tasks')
    group_b = parser.add_argument_group('Arguments Passed to D-SCRIPT Tasks')
    group_a.add_argument(
        "--proteins",
        help="File with protein IDs for which to predict all pairs, one per line; specify one of proteins or pairs",
        required=False,
    )
    group_a.add_argument(
        "--pairs",
        help="File with candidate protein pairs to predict, one pair per line; specify one of proteins or pairs",
        required=False,
    )
    group_b.add_argument(
        "--model",
        help="Pretrained Model. If this is a `.sav` or `.pt` file, it will be loaded. Otherwise, we will try to load `[model]` from HuggingFace hub [default: samsl/topsy_turvy_human_v1]",
        default="samsl/topsy_turvy_human_v1",
    )
    group_b.add_argument(
        "--embeddings",
        help="h5 file with (a superset of) pre-embedded sequences. Generate with dscript embed.",
        required=True,
    )
    group_b.add_argument(
        "--foldseek_fasta",
        help="""3di sequences in .fasta format. Can be generated using `dscript extract-3di.
        Default is None. If provided, TT3D will be run, otherwise default D-SCRIPT/TT will be run.
        """,
        default=None,
    )
    group_a.add_argument("-o", "--workdir", help="Directory for intermediate and output files", required=True)
    group_b.add_argument(
        "-d",
        "--device",
        type=str,
        default="all",
        help="Compute device to use. Options: 'cpu', 'all' (all GPUs), or GPU index (0, 1, 2, etc.). To use specific GPUs, set CUDA_VISIBLE_DEVICES beforehand and use 'all'. [default: all]",
    )
    group_b.add_argument(
        "--store_cmaps",
        action="store_true",
        help="Store contact maps for predicted pairs above `--thresh` in an h5 file",
    )
    group_b.add_argument(
        "--thresh",
        type=float,
        default=0.5,
        help="Positive prediction threshold - used to store contact maps and predictions in a separate file. [default: 0.5]",
    )
    group_b.add_argument(
        "--load_proc",
        type=int,
        default=16,
        help="Number of processes to use when loading embeddings (-1 = # of available CPUs, default=16). Because loading is IO-bound, values larger that the # of CPUs are allowed.",
    )
    group_b.add_argument(
        "--blocks",
        type=int,
        default=1,
        help="Number of equal-sized blocks to split proteins into. In the multi-block case, maximum (embedding) memory usage should be 3 blocks' worth. When multiple GPUs are used, memory usage may briefly be higher when different GPUs are working on tasks from different blocks. And, small blocks may lead to occasional brief hangs with multiple GPUs. Default 1.",
    )
    group_b.add_argument(
        "--sparse_loading",
        action="store_true",
        help="Load only the proteins required from each block, but do not reuse loaded blocks in memory. Recommended when predicting with many blocks on sparse pairs, such that many pairs of blocks might contain no pairs of proteins of interest. Only available when blocks > 1 and pairs specified. Maximum (embedding) memory usage with this option is 4 blocks' worth.",
    )
    group_a.add_argument(
        "--split_blocks",
        type=int,
        default=8,
        help="Number of blocks to use when splitting tasks. Will split into (SB/2)^2 tasks, e.g. 6-> 9, 8->16, each with 3(n/SB)^2 protein pairs. Must be even."
    )
    return parser


def main(args):
    """
    Split prediction tasks into (blocks/2)^2 tasks, for blocks even, where each task has (3*(n/blocks)^2) pairs to predict on

    :meta private:
    """

    # Validate and update paths
    logFile = None
    num_blocks = args.split_blocks
    if num_blocks < 1 or num_blocks %2 == 1:
        num_blocks = max(num_blocks + 1, 4)
        log(f"Can only split based on positive, even # of blocks, incrementing to {num_blocks}", file=logFile, print_also=True)

    modelPath = args.model
    if modelPath.endswith(".sav") or modelPath.endswith(".pt"):
        modelPath = os.path.abspath(modelPath)
        if os.path.isfile(modelPath):
            log(
                f"Will load model locally from {modelPath}", file=logFile, print_also=True
            )
        else:
            log(f"Local model {modelPath} not found; will use this path anyways", file=logFile, print_also=True)
            
    outDir = os.path.abspath(args.workdir)
    if not os.path.exists(outDir):
        log(f"Creating working directory {outDir}", file=logFile, print_also=True)
        os.makedirs(outDir)

    embedFile = args.embeddings
    if os.path.isfile(embedFile):
        embedFile = os.path.abspath(embedFile)
    else:
        log(f"Embeddings file {embedFile} not found; will use this path anyways", file=logFile, print_also=True)

    foldseekFile = args.foldseek_fasta
    if foldseekFile is not None:
        if os.path.isfile(foldseekFile):
            foldseekFile = os.path.abspath(embedFile)
        else:
            log(f"Foldseek file {foldseekFile} not found; will use this path anyways", file=logFile, print_also=True)
    
    fixedArgString = f"--model {modelPath} --device {args.device} --thresh {args.thresh} --load_proc {args.load_proc}"
    if args.store_cmaps:
        fixedArgString += " --store_cmaps"
    allArgString = f"--embeddings {embedFile} --blocks {args.blocks}"
    if args.sparse_loading:
        allArgString += " --sparse_loading"
    if foldseekFile is not None:
        allArgString += f" --foldseek_fasta {foldseekFile}"

    if args.proteins is None == args.pairs is None:
        log(
            "Please specify exactly one of proteins and pairs.",
            file=logFile,
            print_also=True,
        )
        logFile.close()
        sys.exit(2)

    # Load Proteins - all just copied from predict_block
    all_pairs = args.proteins is not None
    if all_pairs:
        tsvPath = args.proteins
        biparArgString = f"--embedA {embedFile} --blocksA {ceil(args.blocks/2)} --blocksB {args.blocks}"
        if foldseekFile is not None:
            biparArgString += f" --foldseekA {foldseekFile}"
    elif args.pairs is not None:
        tsvPath = args.pairs
    else:
        log(
            "One of --proteins and --pairs must be specified.",
            file=logFile,
            print_also=True,
        )
        logFile.close()
        sys.exit(4)
    try:
        log(
            f"Loading {'' if all_pairs else 'pairs of '}protein IDs from {tsvPath}",
            file=logFile,
            print_also=True,
        )
        with open(tsvPath) as f:
            tsv_lines = [line.strip() for line in f if line and not line.isspace()]
    except FileNotFoundError:
        log(f"Proteins / Pairs file {tsvPath} not found", file=logFile, print_also=True)
        logFile.close()
        sys.exit(4)

    if all_pairs:
        all_prots = tsv_lines
        n_prots = len(all_prots)
    # Process a list of pairs into a binary matrix. Not asymptotically efficient for sparse pairs.
    else:
        # Built the data structures we need jointly
        # Also, preserve order of proteins in order of first encounter
        pairs0 = []
        pairs1 = []
        all_prots = []
        prot_to_idx = {}
        for pair in tsv_lines:
            p0, p1 = pair.split("\t")[:2]
            if p0 in prot_to_idx:
                i0 = prot_to_idx[p0]
            else:
                i0 = len(all_prots)
                prot_to_idx[p0] = i0
                all_prots.append(p0)
            if p1 in prot_to_idx:
                i1 = prot_to_idx[p1]
            else:
                i1 = len(all_prots)
                prot_to_idx[p1] = i1
                all_prots.append(p1)
            pairs0.append(i0)
            pairs1.append(i1)
        n_prots = len(all_prots)
        pairs_bool = np.zeros((n_prots, n_prots), dtype=np.bool_)
        pairs_bool[pairs0, pairs1] = 1
        pairs_bool[pairs1, pairs0] = 1
        pairs_bool = np.triu(pairs_bool)  # Makes a copy

    block_size = ceil(n_prots / num_blocks) #Refers to blocks used for splitting here
    def get_bounds(block):
        start = block * block_size
        end = min(start + block_size, n_prots)
        return (start, end)  # all_prots[start:end]

    n_tasks = (num_blocks//2)**2

    def write_self_block(block, script, task_num):
        start1, end1 = get_bounds(block)
        start2, end2 = get_bounds(block+1)
        outfile = os.path.join(outDir, f"predictions_task_{task_num}")
        if all_pairs:
            pfile1 = os.path.join(outDir,f"proteins_group_{block+1}.txt")
            pfile2 = os.path.join(outDir,f"proteins_group_{block+2}.txt")
            with open(pfile1, "w") as prot_file:
                prot_file.write("\n".join(all_prots[start1:end1]))
                prot_file.write("\n")
            with open(pfile2, "w") as prot_file:
                prot_file.write("\n".join(all_prots[start2:end2]))
                prot_file.write("\n")
            print("dscript predict", fixedArgString, allArgString, "--proteins", pfile1, pfile2, "--outfile", outfile, file=script)
        else:
            pfile = os.path.join(outDir,f"pairs_task_{task_num}.txt")
            #Note assumption that blocks are contiguous
            with open(pfile, "w") as pairs_file:
                for i0 in range(start1, end2):
                    for i1 in range(i0+1, end2):
                        if pairs_bool[i0, i1]: #Perhpas not the most efficient
                            pairs_file.write(all_prots[i0] + "\t" + all_prots[i1] + "\n")
            print("dscript predict", fixedArgString, allArgString, "--pairs", pfile, "--outfile", outfile, file=script)
        return task_num + 1

    def write_block(block1, block2, script, task_num):
        outfile = os.path.join(outDir, f"predictions_task_{task_num}")
        if all_pairs:
            pfile1 = os.path.join(outDir,f"proteins_group_{block1+1}.txt")
            pfile2 = os.path.join(outDir,f"proteins_group_{block2+1}.txt")
            pfile3 = os.path.join(outDir,f"proteins_group_{block2+2}.txt")
            print("dscript predict_bipartite", fixedArgString, biparArgString, "--protA", pfile1, "--protB", pfile2, pfile3, "--outfile", outfile, file=script)
        else: #Prep pairs file - will lead to perhaps slightly less-efficient outcome when predicting
            pfile = os.path.join(outDir,f"pairs_task_{task_num}.txt")
            start1, end1 = get_bounds(block1)
            start2, _ = get_bounds(block2)
            _, end3 = get_bounds(block2+1)
            #Note assumption that blocks are contiguous
            with open(pfile, "w") as pairs_file:
                for i0 in range(start1, end1):
                    for i1 in range(start2, end3):
                        if pairs_bool[i0, i1]: #Perhpas not the most efficient
                            pairs_file.write(all_prots[i0] + "\t" + all_prots[i1] + "\n")
            print("dscript predict", fixedArgString, allArgString, "--pairs", pfile, "--outfile", outfile, file=script)
        return task_num + 1

    task_num=1
    script = open(os.path.join(outDir, f"dscript_{n_tasks}_tasks.sh"), "w")
    for i in range(num_blocks):
        #Create block file
        if i % 2 == 0:
            # Do self block
            task_num = write_self_block(i, script, task_num)
            for j in range(i + 2, num_blocks, 2):  # Move up other blocks
                task_num = write_block(i, j, script, task_num)
        else:
            for j in range(i+1, num_blocks, 2):
                task_num = write_block(i, j, script, task_num)
    script.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
