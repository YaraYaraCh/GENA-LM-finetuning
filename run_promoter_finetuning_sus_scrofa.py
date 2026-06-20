#!/usr/bin/env python3
"""
Finetuning GENA-LM on Sus scrofa promoters.
GPU version.
"""

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef

from lm_experiments_tools import Trainer, TrainerArgs, get_optimizer
from lm_experiments_tools.utils import get_cls_by_name, collect_run_configuration
from lm_experiments_tools.utils import get_git_diff
import lm_experiments_tools.optimizers as optimizers

from sus_scrofa_dataset import SusScrofaPromoterDataset

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = HfArgumentParser(TrainerArgs)
parser.add_argument('--data_path', type=str)
parser.add_argument('--valid_data_path', type=str)
parser.add_argument('--test_data_path', type=str)
parser.add_argument('--validate_only', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--input_seq_len', type=int, default=2000)
parser.add_argument('--data_n_workers', type=int, default=2)
parser.add_argument('--model_cfg', type=str)
parser.add_argument('--model_cls', type=str,
                    default='src.gena_lm.modeling_bert:BertForSequenceClassification')
parser.add_argument('--tokenizer', type=str, default=None)
parser.add_argument('--bpe_dropout', type=float, default=None)
parser.add_argument('--optimizer', type=str, default='AdamW')
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--scale_parameter', action='store_true', default=False)
parser.add_argument('--relative_step', action='store_true', default=False)
parser.add_argument('--warmup_init', action='store_true', default=False)
parser.add_argument('--body_lr_multiplier', type=float, default=1.0)


if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Определяем устройство ──────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f'Running on GPU: {torch.cuda.get_device_name(0)}')
        logger.info(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    else:
        device = torch.device('cpu')
        logger.info('CUDA not available, running on CPU')

    if args.model_path is not None:
        model_path = Path(args.model_path)
        if not model_path.exists():
            model_path.mkdir(parents=True)
        args_dict = collect_run_configuration(args)
        json.dump(args_dict, open(model_path / 'config.json', 'w'), indent=4)
        open(model_path / 'git.diff', 'w').write(get_git_diff())

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # ── Датасеты ───────────────────────────────────────────────────────────
    logger.info('Preparing training data from: ' + str(args.data_path))
    train_dataset = SusScrofaPromoterDataset(
        h5_path=args.data_path,
        max_seq_len=args.input_seq_len,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.data_n_workers,
        pin_memory=True,   # быстрее перенос CPU→GPU
    )

    valid_dataloader = None
    if args.valid_data_path:
        logger.info('Preparing validation data from: ' + str(args.valid_data_path))
        valid_dataset = SusScrofaPromoterDataset(
            h5_path=args.valid_data_path,
            max_seq_len=args.input_seq_len,
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.data_n_workers,
            pin_memory=True,
        )
        if args.valid_interval is None:
            args.valid_interval = args.log_interval
    else:
        logger.info('No validation data is used.')

    test_dataloader = None
    if args.test_data_path:
        logger.info('Preparing test data from: ' + str(args.test_data_path))
        test_dataset = SusScrofaPromoterDataset(
            h5_path=args.test_data_path,
            max_seq_len=args.input_seq_len,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.data_n_workers,
            pin_memory=True,
        )

    # ── Модель ─────────────────────────────────────────────────────────────
    model_cfg = AutoConfig.from_pretrained(args.model_cfg)
    model_cfg.num_labels = 2
    model_cls = get_cls_by_name(args.model_cls)
    logger.info('Using model class: ' + str(model_cls))
    model = model_cls(config=model_cfg)

    if args.init_checkpoint:
        logger.info('Loading checkpoint from: ' + str(args.init_checkpoint))
        # Загружаем сразу на нужное устройство
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.info('Missing keys: ' + str(missing[:3]))
        if unexpected:
            logger.info('Unexpected keys: ' + str(unexpected[:3]))

    # Переносим модель на GPU
    model = model.to(device)
    logger.info(f'Model moved to {device}')

    # ── Оптимизатор ────────────────────────────────────────────────────────
    optimizer_cls = get_optimizer(args.optimizer)
    if optimizer_cls is None:
        raise RuntimeError(args.optimizer + ' not found')

    if args.body_lr_multiplier == 1.0:
        optimizer = optimizer_cls(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optimizer_cls(
            [
                {'params': model.bert.parameters(), 'lr': args.lr * args.body_lr_multiplier},
                {'params': model.classifier.parameters()},
            ],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    # ── Метрики ────────────────────────────────────────────────────────────
    def keep_for_metrics_fn(batch, output):
        return {
            'labels':      batch['labels'],
            'predictions': torch.argmax(output['logits'].detach(), dim=-1),
        }

    def metrics_fn(data):
        y = data['labels']
        p = data['predictions']
        # Если тензоры — конвертируем в numpy
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        if isinstance(p, torch.Tensor):
            p = p.cpu().numpy()
        return {
            'accuracy':  float((p == y).sum()) / len(y),
            'f1':        f1_score(y, p),
            'f1_macro':  f1_score(y, p, average='macro'),
            'precision': precision_score(y, p),
            'recall':    recall_score(y, p),
            'mcc':       matthews_corrcoef(y, p),
        }

    # ── Тренер ─────────────────────────────────────────────────────────────
    trainer = Trainer(
        args, model, optimizer,
        train_dataloader, valid_dataloader,
        keep_for_metrics_fn=keep_for_metrics_fn,
        metrics_fn=metrics_fn,
    )

    if not args.validate_only:
        trainer.train()
        if args.save_best:
            best_model_path = str(Path(args.model_path) / 'model_best.pth')
            logger.info('Loading best model from: ' + best_model_path)
            trainer.load(best_model_path)
        if valid_dataloader:
            logger.info('Running validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=False)
        if test_dataloader:
            logger.info('Running validation on test data:')
            trainer.validate(test_dataloader, split='test', write_tb=True)
    else:
        logger.info('Running validation on train set:')
        trainer.validate(train_dataloader, write_tb=False)
        if valid_dataloader:
            trainer.validate(valid_dataloader, write_tb=False)
        if test_dataloader:
            trainer.validate(test_dataloader, split='test', write_tb=False)