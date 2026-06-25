#!/usr/bin/env python3
"""
Генератор HDF5 датасета промоторов для генома человека (GCF_000001405.40).

Обходит все варианты разбиения в ~/notebooks/burek/<variant>/genes/GCF_000001405.40/
и создаёт train/valid/test HDF5 файлы в ~/human/hdf5/<variant>/.

Использование:
    python generator_human.py
    python generator_human.py --variants blastp_enformer blastp_borzoi
    python generator_human.py --fasta ~/human/GCF_000001405.40_genomic.fna
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer

# ── Константы промотора ───────────────────────────────────────────────────────
PROMOTER_UP   = 1000
PROMOTER_DOWN = 999
WINDOW_SIZE   = 2000
ANTI_SHIFT    = 20000
STEP          = 10

HOME = os.path.expanduser("~")

# ── Пути по умолчанию ─────────────────────────────────────────────────────────
DEFAULT_FASTA       = f"{HOME}/human/GCF_000001405.40_genomic.fna"
DEFAULT_BUREK_DIR   = f"{HOME}/notebooks/burek"
DEFAULT_RESULTS_DIR = f"{HOME}/human/hdf5"
DEFAULT_TOKENIZER   = f"{HOME}/GENA_LM/model_checkpoint"
GCF_ID              = "GCF_000001405.40"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate HDF5 promoter datasets for human genome across all burek split variants"
    )
    parser.add_argument("--fasta",        default=DEFAULT_FASTA,
                        help=f"Путь к FASTA генома человека (default: {DEFAULT_FASTA})")
    parser.add_argument("--burek_dir",    default=DEFAULT_BUREK_DIR,
                        help=f"Папка с вариантами разбиения (default: {DEFAULT_BUREK_DIR})")
    parser.add_argument("--results_dir",  default=DEFAULT_RESULTS_DIR,
                        help=f"Папка для HDF5 результатов (default: {DEFAULT_RESULTS_DIR})")
    parser.add_argument("--tokenizer",    default=DEFAULT_TOKENIZER,
                        help=f"Путь к токенизатору GENA-LM (default: {DEFAULT_TOKENIZER})")
    parser.add_argument("--variants",     nargs="+", default=None,
                        help="Список вариантов для обработки (по умолчанию все)")
    return parser.parse_args()


# ── Загрузка генома ───────────────────────────────────────────────────────────
def load_genome(fasta_file):
    print(f"  Загружаю геном: {fasta_file}")
    print("  (это может занять несколько минут для генома человека ~3 ГБ)")
    genome = {}
    for rec in SeqIO.parse(fasta_file, "fasta"):
        genome[rec.id] = str(rec.seq).upper()
    print(f"  Загружено {len(genome)} хромосом/контигов")
    return genome


# ── Загрузка TSV ──────────────────────────────────────────────────────────────
def load_genes_tsv(tsv_file):
    df = pd.read_csv(tsv_file, sep="\t")
    genes = []
    for _, row in df.iterrows():
        genes.append({
            "gene_id":    str(row["gene_id"]),
            "strand":     str(row["strand"]),
            "chromosome": str(row["chromosome"]),
            "tss":        int(row["TSS"]),
            "tes":        int(row["TES"]),
            "split":      str(row["split"]),
        })
    counts = df["split"].value_counts()
    print(f"  Генов в TSV: {len(genes)} | {dict(counts)}")
    return genes


# ── Промотор ──────────────────────────────────────────────────────────────────
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
    return start, window_size - len(part1), part1 + part2


def find_antipromoter(chrom_seq, tss, promoter_coords):
    genome_len      = len(chrom_seq)
    candidate_start = (tss + ANTI_SHIFT) % genome_len
    while True:
        a_start, a_end, candidate_seq = get_window_with_wrap(
            chrom_seq, candidate_start, WINDOW_SIZE)
        overlap = any(
            not (a_end <= p_start or a_start >= p_end)
            for p_start, p_end in promoter_coords
        )
        if not overlap:
            return a_start, a_end, candidate_seq
        candidate_start = (candidate_start + STEP) % genome_len


# ── Токенизация ───────────────────────────────────────────────────────────────
def tokenize_sequence(seq, tokenizer):
    tokens = tokenizer(
        seq,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=WINDOW_SIZE,
    )
    return tokens["input_ids"], tokens["attention_mask"]


# ── Запись в HDF5 ─────────────────────────────────────────────────────────────
def write_group(h5f, group_name, token_ids, attn_mask,
                gene_id, strand, coords, seq_type):
    grp = h5f.create_group(group_name)
    grp.attrs["sample"]  = group_name
    grp.attrs["type"]    = seq_type
    grp.attrs["Gene_ID"] = gene_id
    grp.attrs["strand"]  = strand
    grp.attrs["coords"]  = coords
    grp.attrs["species"] = GCF_ID
    grp.create_dataset("input_ids",      data=token_ids)
    grp.create_dataset("attention_mask", data=attn_mask)


# ── Обработка одного варианта разбиения ───────────────────────────────────────
def process_variant(variant_name, tsv_path, genome, tokenizer, out_dir):
    print(f"\n{'='*60}")
    print(f"  ВАРИАНТ: {variant_name}")
    print(f"{'='*60}")

    genes = load_genes_tsv(tsv_path)

    # Собираем координаты промоторов для поиска антипромоторов
    promoter_dict = {}
    for g in genes:
        if g["chromosome"] not in genome:
            continue
        start, end, _ = get_promoter(genome[g["chromosome"]], g["tss"])
        promoter_dict.setdefault(g["chromosome"], []).append((start, end))

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"{variant_name}_human_promoters_train.h5"
    valid_path = out_dir / f"{variant_name}_human_promoters_valid.h5"
    test_path  = out_dir / f"{variant_name}_human_promoters_test.h5"

    counters = {"train": 0, "valid": 0, "test": 0, "skipped": 0}

    with h5py.File(train_path, "w") as h5_train, \
         h5py.File(valid_path, "w") as h5_valid, \
         h5py.File(test_path,  "w") as h5_test:

        h5_map = {"train": h5_train, "valid": h5_valid, "test": h5_test}

        for g in tqdm(genes, desc=f"  {variant_name}"):
            chrom = g["chromosome"]
            if chrom not in genome:
                counters["skipped"] += 1
                continue

            split = g["split"]
            if split not in h5_map:
                counters["skipped"] += 1
                continue

            h5f    = h5_map[split]
            seq    = genome[chrom]
            tss    = g["tss"]
            strand = g["strand"]
            idx    = counters[split]

            # ПРОМОТОР
            p_start, p_end, promoter_seq = get_promoter(seq, tss)
            token_ids, attn_mask = tokenize_sequence(promoter_seq, tokenizer)
            write_group(h5f, f"{idx}_pos", token_ids, attn_mask,
                        g["gene_id"], strand,
                        f"{chrom}:{p_start}-{p_end}", "promoter")

            # АНТИПРОМОТОР
            a_start, a_end, anti_seq = find_antipromoter(
                seq, tss, promoter_dict.get(chrom, []))
            anti_ids, anti_mask = tokenize_sequence(anti_seq, tokenizer)
            write_group(h5f, f"{idx}_neg", anti_ids, anti_mask,
                        g["gene_id"], strand,
                        f"{chrom}:{a_start}-{a_end}", "antipromoter")

            counters[split] += 1

    print(f"  train={counters['train']*2} | valid={counters['valid']*2} | "
          f"test={counters['test']*2} | пропущено={counters['skipped']}")
    print(f"  Сохранено в: {out_dir}")
    return counters


def main():
    args = parse_args()

    # Проверяем что FASTA скачана
    if not os.path.exists(args.fasta):
        print(f"❌ FASTA не найдена: {args.fasta}")
        print("   Дождись окончания скачивания и распаковки!")
        return

    # Загружаем токенизатор один раз
    print("Загружаю токенизатор...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    print(f"  Размер словаря: {tokenizer.vocab_size}")

    # Загружаем геном один раз для всех вариантов
    genome = load_genome(args.fasta)

    # Определяем список вариантов
    burek_dir = Path(args.burek_dir)
    if args.variants:
        variants = args.variants
    else:
        variants = sorted([
            d.name for d in burek_dir.iterdir()
            if d.is_dir() and d.name != "results"
            and (d / "genes" / GCF_ID / f"{GCF_ID}_genes.tsv").exists()
        ])

    print(f"\nНайдено вариантов с данными для {GCF_ID}: {len(variants)}")
    print(f"Варианты: {variants}")

    # Обрабатываем каждый вариант
    all_stats = {}
    for variant_name in variants:
        tsv_path = burek_dir / variant_name / "genes" / GCF_ID / f"{GCF_ID}_genes.tsv"
        if not tsv_path.exists():
            print(f"\n⚠️  TSV не найден для {variant_name}: {tsv_path}")
            continue
        out_dir = Path(args.results_dir) / variant_name
        stats = process_variant(variant_name, str(tsv_path), genome, tokenizer, out_dir)
        all_stats[variant_name] = stats

    # Итог
    print(f"\n{'='*60}")
    print("ИТОГО:")
    print(f"{'='*60}")
    for v, s in all_stats.items():
        print(f"  {v:20s}: train={s['train']*2:6d} | "
              f"valid={s['valid']*2:6d} | test={s['test']*2:6d}")
    print(f"\nВсе файлы сохранены в: {args.results_dir}")


if __name__ == "__main__":
    main()
