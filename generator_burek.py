#!/usr/bin/env python3
"""
Генератор HDF5 датасетов для структуры burek/.

Обходит все варианты разбиения (blastp_borzoi, cactus_random и т.д.),
внутри каждого варианта читает TSV файлы для всех геномов и создаёт
объединённые train/valid/test HDF5 файлы для каждого варианта.

Структура входных данных:
  ~/GENA_LM/burek/<variant>/genes/<GCF_id>/<GCF_id>_genes.tsv

Структура выходных данных:
  ~/GENA_LM/burek/results/<variant>/<variant>_promoters_train.h5
  ~/GENA_LM/burek/results/<variant>/<variant>_promoters_valid.h5
  ~/GENA_LM/burek/results/<variant>/<variant>_promoters_test.h5
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
PROMOTER_UP   = 1000   # нуклеотидов вверх от TSS
PROMOTER_DOWN = 999    # нуклеотидов вниз от TSS
WINDOW_SIZE   = 2000   # итоговая длина окна
ANTI_SHIFT    = 20000  # сдвиг от TSS для поиска антипромотора
STEP          = 10     # шаг поиска антипромотора при перекрытии

# ── Соответствие GCF идентификаторов и путей к FASTA ──────────────────────────
# Подгоняем под реальные имена файлов в ~/GENA_LM/genomes/fna/ и ~/GENA_LM/sus_scrofa_raw/
HOME = os.path.expanduser("~")
GCF_TO_FASTA = {
    "GCF_000001635.27": f"{HOME}/GENA_LM/genomes/fna/mus_musculus.fna",
    "GCF_000003025.6":  f"{HOME}/GENA_LM/sus_scrofa_raw/sus_scrofa.fna",
    "GCF_002263795.3":  f"{HOME}/GENA_LM/genomes/fna/bos_taurus.fna",
    "GCF_011100685.1":  f"{HOME}/GENA_LM/genomes/fna/canis_lupus.fna",
    "GCF_016772045.2":  f"{HOME}/GENA_LM/genomes/fna/ovis_aries.fna",
    "GCF_036323735.1":  f"{HOME}/GENA_LM/genomes/fna/rattus_norvegicus.fna",
    "GCF_041296265.1":  f"{HOME}/GENA_LM/genomes/fna/equus_caballus.fna",
    "GCF_049350105.2":  f"{HOME}/GENA_LM/genomes/fna/macaca.fna",
    "GCF_964237555.1":  f"{HOME}/GENA_LM/genomes/fna/oryctolagus_cuniculus.fna",
    # Человек намеренно пропущен — для него FASTA нет
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate HDF5 datasets for all burek split variants"
    )
    parser.add_argument("--burek_dir",      default=f"{HOME}/GENA_LM/burek",
                        help="Корневая папка с вариантами разбиения")
    parser.add_argument("--results_dir",    default=f"{HOME}/GENA_LM/burek/results",
                        help="Папка для сохранения HDF5 файлов")
    parser.add_argument("--tokenizer_path", default=f"{HOME}/GENA_LM/model_checkpoint",
                        help="Путь к токенизатору GENA-LM")
    parser.add_argument("--variants",       nargs="+", default=None,
                        help="Список вариантов для обработки (по умолчанию все)")
    return parser.parse_args()


# ── Загрузка генома ───────────────────────────────────────────────────────────
def load_genome(fasta_file):
    """Читает FASTA и возвращает словарь {chromosome_id: sequence}."""
    genome = {}
    for rec in SeqIO.parse(fasta_file, "fasta"):
        genome[rec.id] = str(rec.seq).upper()
    return genome


# ── Загрузка TSV с генами и разбиением ────────────────────────────────────────
def load_genes_tsv(tsv_file):
    """
    Читает новый формат TSV с колонками:
      gene_id, strand, chromosome, TSS, TES, gene_name, split

    Возвращает список словарей с информацией о каждом гене.
    """
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
    return genes


# ── Вырезание промотора ───────────────────────────────────────────────────────
def get_promoter(seq, tss):
    """Вырезает окно [-PROMOTER_UP, +PROMOTER_DOWN] вокруг TSS."""
    start = max(tss - PROMOTER_UP, 0)
    end   = min(tss + PROMOTER_DOWN + 1, len(seq))
    return start, end, seq[start:end]


# ── Вырезание окна с оборачиванием ────────────────────────────────────────────
def get_window_with_wrap(seq, start, window_size):
    """Вырезает окно, оборачиваясь на конец хромосомы если нужно."""
    genome_len = len(seq)
    if start + window_size <= genome_len:
        return start, start + window_size, seq[start:start + window_size]
    part1 = seq[start:]
    part2 = seq[:window_size - len(part1)]
    return start, window_size - len(part1), part1 + part2


# ── Поиск антипромотора ───────────────────────────────────────────────────────
def find_antipromoter(chrom_seq, tss, promoter_coords):
    """Ищет участок без перекрытия с промоторами, сдвинутый на ANTI_SHIFT."""
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
    """Токенизирует строку ДНК через GENA-LM токенизатор."""
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
                gene_id, strand, coords, seq_type, gcf_id):
    """Записывает один сэмпл в HDF5 группу."""
    grp = h5f.create_group(group_name)
    grp.attrs["sample"]  = group_name
    grp.attrs["type"]    = seq_type
    grp.attrs["Gene_ID"] = gene_id
    grp.attrs["strand"]  = strand
    grp.attrs["coords"]  = coords
    grp.attrs["species"] = gcf_id  # GCF идентификатор вместо названия вида
    grp.create_dataset("input_ids",      data=token_ids)
    grp.create_dataset("attention_mask", data=attn_mask)


# ── Обработка одного генома внутри варианта ───────────────────────────────────
def process_genome(gcf_id, tsv_path, tokenizer,
                   h5_train, h5_valid, h5_test, global_counters):
    """Обрабатывает один геном (GCF) внутри одного варианта разбиения."""
    fasta_path = GCF_TO_FASTA.get(gcf_id)
    if not fasta_path or not os.path.exists(fasta_path):
        print(f"    ⚠️  Пропускаю {gcf_id}: FASTA не найдена")
        return {"train": 0, "valid": 0, "test": 0, "skipped": 0}

    print(f"    Загружаю FASTA: {os.path.basename(fasta_path)}")
    genome = load_genome(fasta_path)
    print(f"      {len(genome)} хромосом")

    genes = load_genes_tsv(tsv_path)
    print(f"    Прочитано генов: {len(genes)}")

    # Собираем координаты всех промоторов для поиска антипромоторов
    promoter_dict = {}
    for g in genes:
        if g["chromosome"] not in genome:
            continue
        start, end, _ = get_promoter(genome[g["chromosome"]], g["tss"])
        promoter_dict.setdefault(g["chromosome"], []).append((start, end))

    h5_map = {"train": h5_train, "valid": h5_valid, "test": h5_test}
    species_counters = {"train": 0, "valid": 0, "test": 0, "skipped": 0}

    for g in tqdm(genes, desc=f"      {gcf_id}", leave=False):
        chrom = g["chromosome"]
        if chrom not in genome:
            species_counters["skipped"] += 1
            continue

        split = g["split"]
        if split not in h5_map:
            species_counters["skipped"] += 1
            continue

        h5f    = h5_map[split]
        seq    = genome[chrom]
        tss    = g["tss"]
        strand = g["strand"]
        idx    = global_counters[split]

        # ПРОМОТОР
        p_start, p_end, promoter_seq = get_promoter(seq, tss)
        token_ids, attn_mask = tokenize_sequence(promoter_seq, tokenizer)
        write_group(h5f, f"{idx}_pos", token_ids, attn_mask,
                    g["gene_id"], strand,
                    f"{chrom}:{p_start}-{p_end}", "promoter", gcf_id)

        # АНТИПРОМОТОР
        a_start, a_end, anti_seq = find_antipromoter(
            seq, tss, promoter_dict[chrom])
        anti_ids, anti_mask = tokenize_sequence(anti_seq, tokenizer)
        write_group(h5f, f"{idx}_neg", anti_ids, anti_mask,
                    g["gene_id"], strand,
                    f"{chrom}:{a_start}-{a_end}", "antipromoter", gcf_id)

        global_counters[split]  += 1
        species_counters[split] += 1

    return species_counters


# ── Обработка одного варианта разбиения ───────────────────────────────────────
def process_variant(variant_name, variant_dir, results_dir, tokenizer):
    """Обрабатывает один вариант (например, blastp_borzoi)."""
    print(f"\n{'='*70}")
    print(f"ВАРИАНТ: {variant_name}")
    print(f"{'='*70}")

    genes_dir = Path(variant_dir) / "genes"
    if not genes_dir.exists():
        print(f"  ❌ Папка genes/ не найдена в {variant_dir}")
        return

    # Создаём папку для результатов
    out_dir = Path(results_dir) / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / f"{variant_name}_promoters_train.h5"
    valid_path = out_dir / f"{variant_name}_promoters_valid.h5"
    test_path  = out_dir / f"{variant_name}_promoters_test.h5"

    global_counters = {"train": 0, "valid": 0, "test": 0}
    all_stats = {}

    # Находим все GCF подпапки и обрабатываем их по очереди
    gcf_dirs = sorted([d for d in genes_dir.iterdir() if d.is_dir()])
    print(f"  Найдено геномов: {len(gcf_dirs)}")

    with h5py.File(train_path, "w") as h5_train, \
         h5py.File(valid_path, "w") as h5_valid, \
         h5py.File(test_path,  "w") as h5_test:

        for gcf_dir in gcf_dirs:
            gcf_id   = gcf_dir.name
            tsv_files = list(gcf_dir.glob("*.tsv"))
            if not tsv_files:
                print(f"  ⚠️  В {gcf_id} нет TSV файлов")
                continue
            tsv_path = tsv_files[0]
            print(f"\n  ── {gcf_id} ──")

            stats = process_genome(gcf_id, str(tsv_path), tokenizer,
                                   h5_train, h5_valid, h5_test,
                                   global_counters)
            all_stats[gcf_id] = stats

    # Итог по варианту
    print(f"\n  Итог для {variant_name}:")
    for gcf_id, stats in all_stats.items():
        print(f"    {gcf_id}: train={stats['train']*2:6d} | "
              f"valid={stats['valid']*2:6d} | test={stats['test']*2:6d}")
    print(f"    {'ВСЕГО':17s}: train={global_counters['train']*2:6d} | "
          f"valid={global_counters['valid']*2:6d} | test={global_counters['test']*2:6d}")
    print(f"  Сохранено в: {out_dir}")


def main():
    args = parse_args()

    # Загружаем токенизатор один раз для всех вариантов
    print("Загружаю токенизатор...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,
                                              trust_remote_code=True)
    print(f"  Размер словаря: {tokenizer.vocab_size}")

    # Определяем список вариантов
    burek_dir = Path(args.burek_dir)
    if args.variants:
        variants = args.variants
    else:
        # Берём все папки кроме results
        variants = sorted([d.name for d in burek_dir.iterdir()
                          if d.is_dir() and d.name != "results"])

    print(f"\nБудут обработаны {len(variants)} вариантов: {variants}")

    for variant_name in variants:
        variant_dir = burek_dir / variant_name
        if not variant_dir.exists():
            print(f"\n❌ Папка {variant_dir} не существует, пропускаю")
            continue
        process_variant(variant_name, str(variant_dir),
                        args.results_dir, tokenizer)

    print(f"\n{'='*70}")
    print("ВСЁ ГОТОВО!")
    print(f"Результаты сохранены в: {args.results_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

