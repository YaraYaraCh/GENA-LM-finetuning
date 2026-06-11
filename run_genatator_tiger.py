from transformers import pipeline
import os

FASTA = os.path.expanduser("~/genomes2/tiger_2_chr.fna")
GFF_OUT = os.path.expanduser("~/genomes2/tiger_2_chr.gff")

print("Загружаем пайплайн...")
pipe = pipeline(
    task="genatator-pipeline",
    model="shmelev/genatator-pipeline",
    trust_remote_code=True,
    device=0,
)

print(f"Запускаем предсказание: {FASTA}")
output_path = pipe(
    FASTA,
    output_gff_path=GFF_OUT,
    gene_finding_use_reverse_complement=True,
    transcript_type_use_reverse_complement=True,
    segmentation_use_reverse_complement=True,
    save_intermediate_files=False,
)

print(f"Готово! GFF записан: {output_path}")
