#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer

PROMOTER_UP = 1000
PROMOTER_DOWN = 999
WINDOW_SIZE = 2000
ANTI_SHIFT = 20000
STEP = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Create promoter/anti-promoter HDF5 dataset from FASTA + GTF + TSV split")
    parser.add_argument("--fasta",          required=True,  help="Path to genome FASTA file")
    parser.add_argument("--gtf",            required=True,  help="Path to GTF annotation file")
    parser.add_argument("--split_tsv",      required=True,  help="Path to TSV file with train/valid/test split")
    parser.add_argument("--find",           choices=["all", "mrna"], default="all")
    parser.add_argument("--tokenizer_path", default=None,   help="Local path to tokenizer")
    parser.add_argument("--model_name",     default="AIRI-Institute/gena-lm-bert-large-t2t")
    parser.add_argument("--output_dir",     default=".",    help="Directory to save HDF5 files")
    parser.add_argument("--output_prefix",  default="promoters", help="Prefix for output files")
    return parser.parse_args()


def load_genome(fasta_file):
    print("Loading genome (may take a few minutes)...")
    genome = {}
    for rec in SeqIO.parse(fasta_file, "fasta"):
        genome[rec.id] = str(rec.seq).upper()
    print(f"  Loaded {len(genome)} chromosomes")
    return genome


def load_split_tsv(tsv_file):
    """Загружает TSV и возвращает словарь gene_id -> split (train/valid/test)"""
    df = pd.read_csv(tsv_file, sep='\t')
    split_dict = {}
    for _, row in df.iterrows():
        split_dict[row['gene_id']] = row['split']
    counts = df['split'].value_counts()
    print(f"  Split counts: {dict(counts)}")
    return split_dict


def parse_gtf(gtf_file, mode):
    transcripts = []
    with open(gtf_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) != 9:
                continue
            chrom, source, feature, start, end, score, strand, frame, attrs = parts
            if feature not in ("transcript", "mRNA"):
                continue
            attr_dict = {}
            for field in attrs.split(";"):
                field = field.strip()
                if field and " " in field:
                    key, value = field.split(" ", 1)
                    attr_dict[key] = value.replace('"', "")
            if mode == "mrna":
                if attr_dict.get("transcript_biotype") not in ("mRNA", "protein_coding"):
                    continue
            transcripts.append({
                "chrom":   chrom,
                "start":   int(start) - 1,
                "end":     int(end),
                "strand":  strand,
                "gene_id": attr_dict.get("gene_id", "unknown")
            })
    return transcripts


def select_one_per_gene(transcripts):
    best = {}
    for tx in transcripts:
        gene_id = tx["gene_id"]
        tss = tx["start"] if tx["strand"] == "+" else tx["end"] - 1
        if gene_id not in best:
            best[gene_id] = tx
        else:
            prev_tss = best[gene_id]["start"] if best[gene_id]["strand"] == "+" else best[gene_id]["end"] - 1
            if tss < prev_tss:
                best[gene_id] = tx
    result = list(best.values())
    print(f"  Transcripts before deduplication : {len(transcripts)}")
    print(f"  Transcripts after  deduplication : {len(result)}")
    return result


def get_promoter(seq, tss):
    start = max(tss - PROMOTER_UP, 0)
    end   = min(tss + PROMOTER_DOWN + 1, len(seq))
    return start, end, seq[start:end]


def get_window_with_wrap(seq, start, window_size):
    genome_len = len(seq)
    if start + window_size <= genome_len:
        return start, start + window_size, seq[start:start + window_size]
    part1 = seq[start:]
    part2 = seq[:window_size - len(part1)]
    return start, (window_size - len(part1)), part1 + part2


def find_antipromoter(chrom_seq, tss, promoter_coords):
    genome_len     = len(chrom_seq)
    candidate_start = tss + ANTI_SHIFT
    if candidate_start >= genome_len:
        candidate_start %= genome_len
    while True:
        a_start, a_end, candidate_seq = get_window_with_wrap(chrom_seq, candidate_start, WINDOW_SIZE)
        overlap = any(
            not (a_end <= p_start or a_start >= p_end)
            for p_start, p_end in promoter_coords)
        if not overlap:
            return a_start, a_end, candidate_seq
        candidate_start = (candidate_start + STEP) % genome_len


def tokenize_sequence(seq, tokenizer):
    """Правильная токенизация: строка целиком, не посимвольно"""
    tokens = tokenizer(
        seq,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=WINDOW_SIZE
    )
    return tokens["input_ids"], tokens["attention_mask"]


def write_group(h5f, group_name, token_ids, attn_mask, gene_id, strand, coords, seq_type):
    grp = h5f.create_group(group_name)
    grp.attrs["sample"]  = group_name
    grp.attrs["type"]    = seq_type
    grp.attrs["Gene_ID"] = gene_id
    grp.attrs["strand"]  = strand
    grp.attrs["coords"]  = coords
    grp.create_dataset("input_ids",      data=token_ids)
    grp.create_dataset("attention_mask", data=attn_mask)


def main():
    args = parse_args()
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Токенизатор ───────────────────────────────────────────────────────
    print("Loading tokenizer...")
    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name,     trust_remote_code=True)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # ── Данные ────────────────────────────────────────────────────────────
    genome      = load_genome(args.fasta)
    print("Parsing GTF...")
    transcripts = parse_gtf(args.gtf, args.find)
    print(f"  Found {len(transcripts)} transcripts")
    transcripts = select_one_per_gene(transcripts)

    print("Loading split TSV...")
    split_dict  = load_split_tsv(args.split_tsv)

    # ── Промоторные координаты (для поиска антипромоторов) ────────────────
    print("Collecting promoter coordinates...")
    promoter_dict = {}
    for tx in transcripts:
        chrom = tx["chrom"]
        if chrom not in genome:
            continue
        tss        = tx["start"] if tx["strand"] == "+" else tx["end"] - 1
        start, end, _ = get_promoter(genome[chrom], tss)
        promoter_dict.setdefault(chrom, []).append((start, end))

    # ── Открываем три HDF5 файла ──────────────────────────────────────────
    train_path = os.path.join(args.output_dir, f"{args.output_prefix}_train.h5")
    valid_path = os.path.join(args.output_dir, f"{args.output_prefix}_valid.h5")
    test_path  = os.path.join(args.output_dir, f"{args.output_prefix}_test.h5")

    counters = {"train": 0, "valid": 0, "test": 0, "skipped": 0}

    print("Creating HDF5 datasets...")
    with h5py.File(train_path, "w") as h5_train, \
         h5py.File(valid_path, "w") as h5_valid, \
         h5py.File(test_path,  "w") as h5_test:

        h5_map = {"train": h5_train, "valid": h5_valid, "test": h5_test}

        for tx in tqdm(transcripts):
            chrom   = tx["chrom"]
            gene_id = tx["gene_id"]

            if chrom not in genome:
                counters["skipped"] += 1
                continue

            # Определяем split для этого гена
            split = split_dict.get(gene_id, None)
            if split not in h5_map:
                counters["skipped"] += 1
                continue

            h5f    = h5_map[split]
            seq    = genome[chrom]
            strand = tx["strand"]
            tss    = tx["start"] if strand == "+" else tx["end"] - 1
            idx    = counters[split]

            # PROMOTER
            p_start, p_end, promoter_seq = get_promoter(seq, tss)
            token_ids, attn_mask = tokenize_sequence(promoter_seq, tokenizer)
            write_group(h5f, f"{idx}_pos", token_ids, attn_mask,
                        gene_id, strand, f"{chrom}:{p_start}-{p_end}", "promoter")

            # ANTIPROMOTER
            a_start, a_end, anti_seq = find_antipromoter(seq, tss, promoter_dict[chrom])
            anti_ids, anti_mask      = tokenize_sequence(anti_seq, tokenizer)
            write_group(h5f, f"{idx}_neg", anti_ids, anti_mask,
                        gene_id, strand, f"{chrom}:{a_start}-{a_end}", "antipromoter")

            counters[split] += 1

    # ── Итог ──────────────────────────────────────────────────────────────
    print("\n=== Done! ===")
    print(f"  Train samples : {counters['train'] * 2}  ({counters['train']} promoters + {counters['train']} antipromoters)")
    print(f"  Valid samples : {counters['valid'] * 2}")
    print(f"  Test  samples : {counters['test']  * 2}")
    print(f"  Skipped genes : {counters['skipped']}")
    print(f"\nFiles saved to: {args.output_dir}")
    print(f"  {train_path}")
    print(f"  {valid_path}")
    print(f"  {test_path}")


if __name__ == "__main__":
    main()
