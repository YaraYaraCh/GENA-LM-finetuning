#!/usr/bin/env python3
"""
Finetuning GENA-LM на промоторах 9 видов животных.
Поддерживает один или несколько геномов через запятую в --data_path.
Логирование в TensorBoard (отдельные папки train/ и valid/).
"""

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig, get_linear_schedule_with_warmup
from sklearn.metrics import (f1_score, matthews_corrcoef,
                             precision_score, recall_score)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── Датасет ───────────────────────────────────────────────────────────────────
class PromoterDataset(Dataset):
    """
    Читает HDF5 файл с промоторами/антипромоторами.
    Работает для любого вида — формат файлов одинаковый.
    """
    def __init__(self, h5_path: str, max_seq_len: int = 512):
        self.h5_path     = str(h5_path)
        self.max_seq_len = max_seq_len
        with h5py.File(self.h5_path, 'r') as f:
            # Сортируем ключи: сначала по числовому индексу, потом по pos/neg
            self.keys = sorted(
                f.keys(),
                key=lambda x: (int(x.split('_')[0]), x.split('_')[1])
            )
        self.length = len(self.keys)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        key = self.keys[idx]
        with h5py.File(self.h5_path, 'r') as f:
            grp            = f[key]
            input_ids      = grp['input_ids'][0].astype(np.int64)
            attention_mask = grp['attention_mask'][0].astype(np.int64)
            # Метка: 1 = промотор, 0 = антипромотор
            label = 1 if grp.attrs.get('type', '') == 'promoter' else 0

        return {
            'input_ids':      torch.tensor(input_ids[:self.max_seq_len],      dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask[:self.max_seq_len], dtype=torch.long),
            'labels':         torch.tensor(label, dtype=torch.long),
        }


# ── Вспомогательные функции ───────────────────────────────────────────────────
def build_dataset(paths_str: str, max_seq_len: int) -> Dataset:
    """
    Принимает строку с путями через запятую и возвращает объединённый датасет.
    Например: 'sus_scrofa/train.h5,bos_taurus/train.h5'
    """
    paths    = [p.strip() for p in paths_str.split(',')]
    datasets = [PromoterDataset(p, max_seq_len=max_seq_len) for p in paths]
    if len(datasets) == 1:
        return datasets[0]
    combined = ConcatDataset(datasets)
    logger.info(f'  Объединено {len(datasets)} геномов → {len(combined)} сэмплов')
    return combined


def get_model_cls(model_cls_str: str):
    """Загружает класс модели по строке вида 'module:ClassName'."""
    import importlib
    module_name, cls_name = model_cls_str.split(':')
    return getattr(importlib.import_module(module_name), cls_name)


def compute_metrics(labels, predictions) -> dict:
    y, p = np.array(labels), np.array(predictions)
    return {
        'accuracy':  float((p == y).sum()) / len(y),
        'f1':        f1_score(y, p, zero_division=0),
        'f1_macro':  f1_score(y, p, average='macro', zero_division=0),
        'precision': precision_score(y, p, zero_division=0),
        'recall':    recall_score(y, p, zero_division=0),
        'mcc':       matthews_corrcoef(y, p),
    }


@torch.no_grad()
def evaluate(model, dataloader, device, seq_len, writer, global_step, split='valid', use_bf16=False) -> dict:
    """Считает метрики на dataloader и пишет их в TensorBoard."""
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model.eval()
    all_labels, all_preds = [], []
    total_loss = 0.0

    for batch in dataloader:
        input_ids      = batch['input_ids'][:, :seq_len].to(device)
        attention_mask = batch['attention_mask'][:, :seq_len].to(device)
        labels         = batch['labels'].to(device)

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_bf16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        preds       = torch.argmax(outputs.logits, dim=-1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    metrics         = compute_metrics(all_labels, all_preds)
    metrics['loss'] = total_loss / len(dataloader)

    # Записываем каждую метрику в TensorBoard
    for k, v in metrics.items():
        writer.add_scalar(k, v, global_step)

    logger.info(f'[{split}] step={global_step} | ' +
                ' | '.join(f'{k}: {v:.4f}' for k, v in metrics.items()))
    model.train()
    return metrics


# ── Аргументы командной строки ────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description='Finetune GENA-LM on promoter/antipromoter classification'
    )
    # Пути к данным — несколько геномов через запятую
    parser.add_argument('--data_path',       type=str, required=True)
    parser.add_argument('--valid_data_path', type=str, required=True)
    parser.add_argument('--test_data_path',  type=str, required=True)

    # Пути к модели
    parser.add_argument('--model_path',      type=str, required=True)
    parser.add_argument('--tokenizer',       type=str, required=True)
    parser.add_argument('--model_cfg',       type=str, required=True)
    parser.add_argument('--init_checkpoint', type=str, default=None)
    parser.add_argument('--model_cls',       type=str,
                        default='src.gena_lm.modeling_bert:BertForSequenceClassification')

    # Гиперпараметры обучения
    parser.add_argument('--input_seq_len',   type=int,   default=512)
    parser.add_argument('--batch_size',      type=int,   default=8)
    parser.add_argument('--iters',           type=int,   default=200000)
    parser.add_argument('--lr',              type=float, default=1e-5)
    parser.add_argument('--weight_decay',    type=float, default=0.01)
    parser.add_argument('--num_warmup_steps',type=int,   default=1000)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--body_lr_multiplier', type=float, default=0.1)
    parser.add_argument('--clip_grad_norm',  type=float, default=1.0)

    # Логирование и остановка
    parser.add_argument('--log_interval',    type=int, default=100)
    parser.add_argument('--valid_interval',  type=int, default=500)
    parser.add_argument('--early_stopping_patience', type=int, default=30)
    parser.add_argument('--data_n_workers',  type=int, default=2)
    parser.add_argument('--seed',            type=int, default=42)
    # Начальный шаг — передай последний шаг предыдущего прогона
    # чтобы TensorBoard рисовал непрерывную кривую через все версии
    parser.add_argument('--start_step',      type=int, default=0)
    return parser.parse_args()


# ── Основная функция ──────────────────────────────────────────────────────────
def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Определяем устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    if device.type == 'cuda':
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
        logger.info(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    # bf16 — вдвое быстрее на RTX 5080 без потери качества
    use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    dtype    = torch.bfloat16 if use_bf16 else torch.float32
    logger.info(f'Mixed precision: {"bf16" if use_bf16 else "fp32"}')

    # Создаём папки и TensorBoard — отдельные папки для train и valid,
    # чтобы они отображались на одном графике в TensorBoard
    model_path   = Path(args.model_path)
    model_path.mkdir(parents=True, exist_ok=True)
    tb_train_dir = model_path / 'tensorboard' / 'train'
    tb_valid_dir = model_path / 'tensorboard' / 'valid'
    tb_train_dir.mkdir(parents=True, exist_ok=True)
    tb_valid_dir.mkdir(parents=True, exist_ok=True)

    writer_train = SummaryWriter(log_dir=str(tb_train_dir))
    writer_valid = SummaryWriter(log_dir=str(tb_valid_dir))
    logger.info(f'TensorBoard: {model_path / "tensorboard"}')

    # Сохраняем конфигурацию запуска
    json.dump(vars(args), open(model_path / 'run_config.json', 'w'), indent=4)

    # Строим датасеты
    logger.info('Загружаю датасеты...')
    train_dataset = build_dataset(args.data_path,       args.input_seq_len)
    valid_dataset = build_dataset(args.valid_data_path, args.input_seq_len)
    test_dataset  = build_dataset(args.test_data_path,  args.input_seq_len)
    logger.info(f'Train: {len(train_dataset)} | Valid: {len(valid_dataset)} | Test: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.data_n_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.data_n_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.data_n_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    # Загружаем модель
    model_cfg             = AutoConfig.from_pretrained(args.model_cfg)
    model_cfg.num_labels  = 2   # бинарная классификация: промотор / антипромотор
    model_cls             = get_model_cls(args.model_cls)
    logger.info(f'Класс модели: {model_cls}')
    model = model_cls(config=model_cfg)

    # Загружаем веса чекпоинта если указан
    if args.init_checkpoint:
        logger.info(f'Загружаю чекпоинт: {args.init_checkpoint}')
        ckpt       = torch.load(args.init_checkpoint, map_location='cpu')
        state_dict = ckpt.get('model_state_dict', ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.info(f'Missing keys (первые 3): {missing[:3]}')
        if unexpected:
            logger.info(f'Unexpected keys (первые 3): {unexpected[:3]}')

    model = model.to(device)

    # Оптимизатор с разными lr для тела BERT и классификационной головы.
    # Тело обучаем медленнее (body_lr_multiplier=0.1), чтобы не разрушить
    # предобученные представления.
    if args.body_lr_multiplier != 1.0:
        optimizer = torch.optim.AdamW([
            {'params': model.bert.parameters(),       'lr': args.lr * args.body_lr_multiplier},
            {'params': model.classifier.parameters(), 'lr': args.lr},
        ], weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Линейный планировщик с warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.iters,
    )

    # ── Тренировочный цикл ────────────────────────────────────────────────
    best_mcc         = -1.0   # MCC от -1 до +1, старт с минимума
    patience_counter = 0
    global_step      = args.start_step  # смещение для непрерывного TensorBoard
    train_iter       = iter(train_loader)
    loss_accum       = 0.0

    model.train()
    optimizer.zero_grad()
    logger.info(f'Начинаю обучение на {args.iters} итераций (с шага {args.start_step})...')

    while global_step < args.start_step + args.iters:
        # Бесконечный итератор — когда датасет заканчивается, начинаем сначала
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch      = next(train_iter)

        input_ids      = batch['input_ids'][:, :args.input_seq_len].to(device)
        attention_mask = batch['attention_mask'][:, :args.input_seq_len].to(device)
        labels         = batch['labels'].to(device)

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_bf16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # Делим loss на gradient_accumulation_steps, чтобы усреднить градиенты
        loss = outputs.loss / args.gradient_accumulation_steps
        loss.backward()
        # Накапливаем реальный (не делённый) loss для логирования
        loss_accum += outputs.loss.item()

        # Обновляем веса только каждые gradient_accumulation_steps шагов
        if (global_step + 1) % args.gradient_accumulation_steps == 0:
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1

        # Логируем train loss
        if global_step % args.log_interval == 0:
            avg_loss   = loss_accum / args.log_interval
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f'Step {global_step}/{args.iters} | loss: {avg_loss:.4f} | lr: {current_lr:.2e}')
            writer_train.add_scalar('loss', avg_loss,   global_step)
            writer_train.add_scalar('lr',   current_lr, global_step)
            loss_accum = 0.0

        # Валидация
        if global_step % args.valid_interval == 0:
            metrics    = evaluate(model, valid_loader, device, args.input_seq_len,
                                  writer_valid, global_step, split='valid', use_bf16=use_bf16)
            current_mcc = metrics['mcc']

            if current_mcc > best_mcc:
                best_mcc         = current_mcc
                patience_counter = 0
                best_path        = model_path / 'model_best.pth'
                torch.save(model.state_dict(), best_path)
                logger.info(f'✅ Новый лучший MCC: {best_mcc:.4f} → сохранено в {best_path}')
            else:
                patience_counter += 1
                logger.info(f'Без улучшений. Patience: {patience_counter}/{args.early_stopping_patience}')
                if patience_counter >= args.early_stopping_patience:
                    logger.info('Early stopping!')
                    break

    # ── Финальная оценка на лучшей модели ────────────────────────────────
    logger.info('Загружаю лучшую модель для финальной оценки...')
    model.load_state_dict(torch.load(model_path / 'model_best.pth', map_location=device))

    logger.info('=== Validation set ===')
    evaluate(model, valid_loader, device, args.input_seq_len,
             writer_valid, global_step, split='valid_final', use_bf16=use_bf16)

    logger.info('=== Test set ===')
    evaluate(model, test_loader, device, args.input_seq_len,
             writer_valid, global_step, split='test', use_bf16=use_bf16)

    writer_train.close()
    writer_valid.close()
    logger.info('Done!')


if __name__ == '__main__':
    main()
