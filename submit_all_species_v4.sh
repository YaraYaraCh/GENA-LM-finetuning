#!/usr/bin/env bash
#SBATCH --job-name=all_species_v4
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/chumanova/GENA_LM/logs/all_species_v4_%j.out
#SBATCH --error=/home/chumanova/GENA_LM/logs/all_species_v4_%j.err

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate gena_lm
export PYTHONPATH="${PYTHONPATH}:/home/chumanova/GENA_LM-main"

cd /home/chumanova/GENA_LM

python ~/notebooks/GENA_LM/run_finetuning_all_gen.py \
    --data_path \
sus_scrofa_new/sus_scrofa_promoters_train.h5,bos_taurus/bos_taurus_promoters_train.h5,canis_lupus/canis_lupus_promoters_train.h5,equus_caballus/equus_caballus_promoters_train.h5,ovis_aries/ovis_aries_promoters_train.h5,oryctolagus_cuniculus/oryctolagus_cuniculus_promoters_train.h5,rattus_norvegicus/rattus_norvegicus_promoters_train.h5,mus_musculus/mus_musculus_promoters_train.h5,macaca/macaca_promoters_train.h5 \
    --valid_data_path \
sus_scrofa_new/sus_scrofa_promoters_valid.h5,bos_taurus/bos_taurus_promoters_valid.h5,canis_lupus/canis_lupus_promoters_valid.h5,equus_caballus/equus_caballus_promoters_valid.h5,ovis_aries/ovis_aries_promoters_valid.h5,oryctolagus_cuniculus/oryctolagus_cuniculus_promoters_valid.h5,rattus_norvegicus/rattus_norvegicus_promoters_valid.h5,mus_musculus/mus_musculus_promoters_valid.h5,macaca/macaca_promoters_valid.h5 \
    --test_data_path \
sus_scrofa_new/sus_scrofa_promoters_test.h5,bos_taurus/bos_taurus_promoters_test.h5,canis_lupus/canis_lupus_promoters_test.h5,equus_caballus/equus_caballus_promoters_test.h5,ovis_aries/ovis_aries_promoters_test.h5,oryctolagus_cuniculus/oryctolagus_cuniculus_promoters_test.h5,rattus_norvegicus/rattus_norvegicus_promoters_test.h5,mus_musculus/mus_musculus_promoters_test.h5,macaca/macaca_promoters_test.h5 \
    --model_path runs/all_species_v4 \
    --init_checkpoint runs/all_species_v2/model_best.pth \
    --tokenizer model_checkpoint \
    --model_cfg model_checkpoint/config.json \
    --model_cls src.gena_lm.modeling_bert:BertForSequenceClassification \
    --input_seq_len 512 \
    --batch_size 8 \
    --iters 100000 \
    --lr 1e-5 \
    --weight_decay 0.01 \
    --num_warmup_steps 500 \
    --gradient_accumulation_steps 16 \
    --body_lr_multiplier 0.1 \
    --clip_grad_norm 1.0 \
    --log_interval 100 \
    --valid_interval 500 \
    --early_stopping_patience 30 \
    --data_n_workers 2 \
    --seed 42
