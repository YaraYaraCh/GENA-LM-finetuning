#!/usr/bin/env python3
"""
Finetuning GENA-LM на промоторах.
Чистый PyTorch без horovod и lm_experiments_tools.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef

from sus_scrofa_dataset import SusScrofaPromoterDataset

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',       type=str, required=True)
    parser.add_argument('--valid_data_path', type=str, required=True)
    parser.add_argument('--test_data_path',  type=str, required=True)
    parser.add_argument('--model_path',      type=str, required=True)
    parser.add_argument('--tokenizer',       type=str, required=True)
    parser.add_argument('--model_cfg',       type=str, required=True)
    parser.add_argument('--init_checkpoint', type=str, default=None)
    parser.add_argument('--model_cls',       type=str,
                        default='src.gena_lm.modeling_bert:BertForSequenceClassification')
    parser.add_argument('--input_seq_len',   type=int, default=512)
    parser.add_argument('--batch_size',      type=int, default=4)
    parser.add_argument('--iters',           type=int, default=3000)
    parser.add_argument('--lr',              type=float, default=1e-4)
    parser.add_argument('--weight_decay',    type=float, default=0.01)
    parser.add_argument('--num_warmup_steps',type=int, default=250)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--body_lr_multiplier', type=float, default=0.1)
    parser.add_argument('--clip_grad_norm',  type=float, default=1.0)
    parser.add_argument('--log_interval',    type=int, default=100)
    parser.add_argument('--valid_interval',  type=int, default=100)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--data_n_workers',  type=int, default=2)
    parser.add_argument('--seed',            type=int, default=42)
    return parser.parse_args()


def get_model_cls(model_cls_str):
    """Загружает класс модели по строке 'module:ClassName'"""
    import importlib
    module_name, cls_name = model_cls_str.split(':')
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def compute_metrics(labels, predictions):
    y = np.array(labels)
    p = np.array(predictions)
    return {
        'accuracy':  float((p == y).sum()) / len(y),
        'f1':        f1_score(y, p, zero_division=0),
        'f1_macro':  f1_score(y, p, average='macro', zero_division=0),
        'precision': precision_score(y, p, zero_division=0),
        'recall':    recall_score(y, p, zero_division=0),
        'mcc':       matthews_corrcoef(y, p),
    }


@torch.no_grad()
def evaluate(model, dataloader, device, split='valid'):
    model.eval()
    all_labels, all_preds = [], []
    total_loss = 0.0

    for batch in dataloader:
        input_ids      = batch['input_ids'][:, :args.input_seq_len].to(device)
        attention_mask = batch['attention_mask'][:, :args.input_seq_len].to(device)
        labels         = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += outputs.loss.item()
        preds = torch.argmax(outputs.logits, dim=-1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds)
    metrics['loss'] = total_loss / len(dataloader)
    logger.info(f'[{split}] ' + ' | '.join(f'{k}: {v:.4f}' for k, v in metrics.items()))
    model.train()
    return metrics


def main():
    global args
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Устройство ────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    if device.type == 'cuda':
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}')

    # ── Папка для сохранения ──────────────────────────────────────────────
    model_path = Path(args.model_path)
    model_path.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(model_path / 'run_config.json', 'w'), indent=4)

    # ── Датасеты ──────────────────────────────────────────────────────────
    logger.info(f'Train data: {args.data_path}')
    train_dataset = SusScrofaPromoterDataset(args.data_path,  max_seq_len=args.input_seq_len)
    valid_dataset = SusScrofaPromoterDataset(args.valid_data_path, max_seq_len=args.input_seq_len)
    test_dataset  = SusScrofaPromoterDataset(args.test_data_path,  max_seq_len=args.input_seq_len)

    logger.info(f'Train: {len(train_dataset)} | Valid: {len(valid_dataset)} | Test: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.data_n_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.data_n_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.data_n_workers, pin_memory=True)

    # ── Модель ────────────────────────────────────────────────────────────
    model_cfg = AutoConfig.from_pretrained(args.model_cfg)
    model_cfg.num_labels = 2
    model_cls = get_model_cls(args.model_cls)
    logger.info(f'Model class: {model_cls}')
    model = model_cls(config=model_cfg)

    if args.init_checkpoint:
        logger.info(f'Loading checkpoint: {args.init_checkpoint}')
        ckpt = torch.load(args.init_checkpoint, map_location='cpu')
        state_dict = ckpt.get('model_state_dict', ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f'Missing keys: {missing[:3]}')
        logger.info(f'Unexpected keys: {unexpected[:3]}')

    model = model.to(device)

    # ── Оптимизатор ───────────────────────────────────────────────────────
    if args.body_lr_multiplier != 1.0:
        optimizer = torch.optim.AdamW([
            {'params': model.bert.parameters(), 'lr': args.lr * args.body_lr_multiplier},
            {'params': model.classifier.parameters(), 'lr': args.lr},
        ], weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.lr, weight_decay=args.weight_decay)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.iters,
    )

    # ── Тренировочный цикл ────────────────────────────────────────────────
    best_f1 = 0.0
    patience_counter = 0
    global_step = 0
    train_iter = iter(train_loader)
    loss_accum = 0.0

    model.train()
    optimizer.zero_grad()

    logger.info(f'Starting training for {args.iters} iterations...')

    while global_step < args.iters:
        # бесконечный итератор по train
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids      = batch['input_ids'][:, :args.input_seq_len].to(device)
        attention_mask = batch['attention_mask'][:, :args.input_seq_len].to(device)
        labels         = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / args.gradient_accumulation_steps
        loss.backward()
        loss_accum += loss.item()

        if (global_step + 1) % args.gradient_accumulation_steps == 0:
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1

        # ── Лог ───────────────────────────────────────────────────────────
        if global_step % args.log_interval == 0:
            logger.info(f'Step {global_step}/{args.iters} | loss: {loss_accum/args.log_interval:.4f} | lr: {scheduler.get_last_lr()[0]:.2e}')
            loss_accum = 0.0

        # ── Валидация ─────────────────────────────────────────────────────
        if global_step % args.valid_interval == 0:
            metrics = evaluate(model, valid_loader, device, split='valid')
            current_f1 = metrics['f1']

            if current_f1 > best_f1:
                best_f1 = current_f1
                patience_counter = 0
                best_path = model_path / 'model_best.pth'
                torch.save(model.state_dict(), best_path)
                logger.info(f'✅ New best F1: {best_f1:.4f} — saved to {best_path}')
            else:
                patience_counter += 1
                logger.info(f'No improvement. Patience: {patience_counter}/{args.early_stopping_patience}')
                if patience_counter >= args.early_stopping_patience:
                    logger.info('Early stopping!')
                    break

    # ── Финальная оценка ──────────────────────────────────────────────────
    logger.info('Loading best model for final evaluation...')
    model.load_state_dict(torch.load(model_path / 'model_best.pth', map_location=device))

    logger.info('=== Validation set ===')
    evaluate(model, valid_loader, device, split='valid')

    logger.info('=== Test set ===')
    evaluate(model, test_loader, device, split='test')

    logger.info('Done!')


if __name__ == '__main__':
    main()
