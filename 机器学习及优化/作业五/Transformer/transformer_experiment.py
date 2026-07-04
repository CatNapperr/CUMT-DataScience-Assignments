# -*- coding: utf-8 -*-
"""
Transformer 文本分类实验：IMDB 情感分类
运行前请先安装依赖：
pip install torch transformers datasets scikit-learn pandas matplotlib tqdm

说明：
1. 默认优先使用 IMDB 数据集；如果下载失败，会自动退回到 SST-2，再退回到二值化 AG News。
2. 为了兼顾普通 GPU / CPU 的可运行性，默认使用子集训练；如需全量训练，可将各类 *_limit 设为 None。
3. 本脚本包含三组实验：
   - 从零实现的 Transformer 编码器文本分类
   - BERT 微调（bert-base-uncased）
   - TF-IDF + LogisticRegression 基线
"""

import os
import re
import sys
import math
import json
import random
import atexit
from datetime import datetime
from dataclasses import dataclass
from collections import Counter
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup


@dataclass
class Config:
    """实验配置类，集中管理超参数与路径。"""
    seed: int = 42
    save_dir: str = "outputs"

    # 数据与预处理
    max_vocab_size: int = 20000
    min_freq: int = 2
    max_length: int = 256
    val_ratio: float = 0.1

    # 为了保证 CPU 也能跑通，默认使用子集；如需全量训练可设为 None
    train_limit: Optional[int] = 12000
    val_limit: Optional[int] = 2000
    test_limit: Optional[int] = 5000

    # 从零实现 Transformer
    batch_size: int = 32
    transformer_epochs: int = 10
    transformer_lr: float = 1e-3
    transformer_weight_decay: float = 1e-4
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ff_dim: int = 256
    dropout: float = 0.1
    early_stop_patience: int = 3

    # BERT 微调
    bert_model_name: str = "bert-base-uncased"
    bert_batch_size: int = 16
    bert_epochs: int = 3
    bert_lr: float = 2e-5

    # DataLoader
    num_workers: int = 0


def set_seed(seed: int) -> None:
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TeeStream:
    """将终端输出同时写入日志文件，保持原有命令行显示不变。"""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file
        self.encoding = getattr(terminal, "encoding", "utf-8")

    def write(self, message: str) -> int:
        self.terminal.write(message)
        self.log_file.write(message)
        return len(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()

    def fileno(self) -> int:
        return self.terminal.fileno()


def setup_training_log(save_dir: str) -> str:
    """开启训练日志，将 stdout 和 stderr 同步写入日志文件。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(save_dir, f"training_{timestamp}.log")
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)

    def cleanup_log() -> None:
        try:
            print(f"\n训练日志已保存：{log_path}")
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()

    atexit.register(cleanup_log)
    print(f"训练日志文件：{log_path}")
    return log_path


def clean_text(text: str) -> str:
    """对原始文本做基础清洗。"""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s'.,!?;:()-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_tokenize(text: str) -> List[str]:
    """使用简单正则分词，适合从零实现模型构建词表。"""
    tokens = re.findall(r"[a-z0-9']+|[.,!?;:()\-]", text)
    return tokens if tokens else ["[UNK]"]


def limit_samples(texts: List[str], labels: List[int], limit: Optional[int], seed: int) -> Tuple[List[str], List[int]]:
    """按类别分层抽样，控制训练规模。"""
    if limit is None or limit >= len(texts):
        return texts, labels
    indices = np.arange(len(texts))
    keep_idx, _ = train_test_split(indices, train_size=limit, stratify=labels, random_state=seed)
    keep_idx = sorted(keep_idx.tolist())
    return [texts[i] for i in keep_idx], [labels[i] for i in keep_idx]


def load_binary_text_dataset(config: Config) -> Tuple[str, List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    加载二分类文本数据。
    返回：
        dataset_name, train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
    """
    dataset_name = ""
    raw_train_texts, raw_train_labels = [], []
    raw_test_texts, raw_test_labels = [], []

    try:
        imdb = load_dataset("imdb")
        dataset_name = "IMDB"
        raw_train_texts = [item["text"] for item in imdb["train"]]
        raw_train_labels = [int(item["label"]) for item in imdb["train"]]
        raw_test_texts = [item["text"] for item in imdb["test"]]
        raw_test_labels = [int(item["label"]) for item in imdb["test"]]
    except Exception as e1:
        print(f"IMDB 下载失败，开始尝试 SST-2。错误信息：{e1}")
        try:
            sst2 = load_dataset("glue", "sst2")
            dataset_name = "SST-2"
            raw_train_texts = [item["sentence"] for item in sst2["train"]]
            raw_train_labels = [int(item["label"]) for item in sst2["train"]]
            raw_test_texts = [item["sentence"] for item in sst2["validation"]]
            raw_test_labels = [int(item["label"]) for item in sst2["validation"]]
        except Exception as e2:
            print(f"SST-2 下载失败，开始尝试 AG News（二值化）。错误信息：{e2}")
            ag_news = load_dataset("ag_news")
            dataset_name = "AG News (binary)"
            raw_train_texts = [item["text"] for item in ag_news["train"]]
            raw_train_labels = [0 if int(item["label"]) < 2 else 1 for item in ag_news["train"]]
            raw_test_texts = [item["text"] for item in ag_news["test"]]
            raw_test_labels = [0 if int(item["label"]) < 2 else 1 for item in ag_news["test"]]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        raw_train_texts,
        raw_train_labels,
        test_size=config.val_ratio,
        stratify=raw_train_labels,
        random_state=config.seed
    )

    train_texts, train_labels = limit_samples(train_texts, train_labels, config.train_limit, config.seed)
    val_texts, val_labels = limit_samples(val_texts, val_labels, config.val_limit, config.seed)
    test_texts, test_labels = limit_samples(raw_test_texts, raw_test_labels, config.test_limit, config.seed)

    return dataset_name, train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def build_vocab(texts: List[str], max_vocab_size: int, min_freq: int) -> Dict[str, int]:
    """根据训练集文本构建词表。"""
    counter = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))

    vocab_tokens = ["[PAD]", "[UNK]"]
    frequent_tokens = [token for token, freq in counter.most_common() if freq >= min_freq]
    vocab_tokens.extend(frequent_tokens[: max_vocab_size - len(vocab_tokens)])

    vocab = {token: idx for idx, token in enumerate(vocab_tokens)}
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """将文本编码为定长 token id 和 attention mask。"""
    pad_id = vocab["[PAD]"]
    unk_id = vocab["[UNK]"]

    tokens = simple_tokenize(text)
    token_ids = [vocab.get(token, unk_id) for token in tokens[:max_length]]
    attention_mask = [1] * len(token_ids)

    if len(token_ids) < max_length:
        pad_len = max_length - len(token_ids)
        token_ids.extend([pad_id] * pad_len)
        attention_mask.extend([0] * pad_len)

    return torch.tensor(token_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)


class ScratchTextDataset(Dataset):
    """从零实现 Transformer 使用的数据集。"""

    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], max_length: int):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask = encode_text(self.texts[idx], self.vocab, self.max_length)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


class BertTextDataset(Dataset):
    """BERT 微调使用的数据集。"""

    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


class PositionalEncoding(nn.Module):
    """手工实现正弦位置编码。"""

    def __init__(self, d_model: int, max_length: int):
        super().__init__()
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class MultiHeadSelfAttention(nn.Module):
    """手工实现多头自注意力机制。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除。"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(context)
        return output


class FeedForwardNetwork(nn.Module):
    """前馈神经网络模块。"""

    def __init__(self, d_model: int, ff_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    """单层 Transformer 编码器，包含注意力、残差连接、层归一化和前馈网络。"""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output = self.attn(x, attention_mask)
        x = self.norm1(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class TransformerClassifier(nn.Module):
    """从零实现的 Transformer 编码器文本分类模型。"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_dim: int,
        max_length: int,
        dropout: float,
        num_classes: int,
        pad_idx: int
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_length)
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(d_model, n_heads, ff_dim, dropout) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, attention_mask)

        # 使用 attention_mask 做掩码平均池化，避免 PAD 对分类结果造成干扰
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        logits = self.classifier(pooled)
        return logits


class EarlyStopping:
    """根据验证集指标执行早停，并自动保存最优权重。"""

    def __init__(self, patience: int, save_path: str, mode: str = "max", delta: float = 0.0):
        self.patience = patience
        self.save_path = save_path
        self.mode = mode
        self.delta = delta
        self.best_score = None
        self.counter = 0

    def step(self, score: float, model: nn.Module) -> Tuple[bool, bool]:
        improved = False

        if self.best_score is None:
            improved = True
        elif self.mode == "max" and score > self.best_score + self.delta:
            improved = True
        elif self.mode == "min" and score < self.best_score - self.delta:
            improved = True

        if improved:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            return False, True

        self.counter += 1
        should_stop = self.counter >= self.patience
        return should_stop, False


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, object]:
    """计算准确率、F1 和混淆矩阵。"""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }


def plot_training_curves(history: Dict[str, List[float]], save_path: str, title: str) -> None:
    """绘制并保存训练过程中的 loss / accuracy 曲线。"""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="o")
    axes[0].set_title(f"{title} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Acc", marker="o")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc", marker="o")
    axes[1].set_title(f"{title} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], save_path: str, title: str) -> None:
    """绘制并保存混淆矩阵图像。"""
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted Label",
        ylabel="True Label",
        title=title
    )

    threshold = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def train_one_epoch_scratch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """训练从零实现的 Transformer 一个 epoch。"""
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []

    for batch in tqdm(loader, desc="Scratch Train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    metrics = compute_metrics(y_true, y_pred)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, metrics["accuracy"], metrics["f1"]


@torch.no_grad()
def evaluate_scratch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, np.ndarray, List[int], List[int]]:
    """评估从零实现的 Transformer。"""
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    for batch in tqdm(loader, desc="Scratch Eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    metrics = compute_metrics(y_true, y_pred)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, metrics["accuracy"], metrics["f1"], metrics["confusion_matrix"], y_true, y_pred


def fit_scratch_transformer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    device: torch.device,
    save_path: str
) -> Dict[str, List[float]]:
    """训练从零实现的 Transformer，并执行验证和早停。"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.transformer_lr,
        weight_decay=config.transformer_weight_decay
    )
    early_stopping = EarlyStopping(config.early_stop_patience, save_path, mode="max")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, config.transformer_epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch_scratch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, _, _, _ = evaluate_scratch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"[Scratch][Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        should_stop, improved = early_stopping.step(val_f1, model)
        if improved:
            print("  验证集 F1 提升，已保存当前最优模型。")
        if should_stop:
            print("  触发早停，结束 Scratch Transformer 训练。")
            break

    model.load_state_dict(torch.load(save_path, map_location=device))
    return history


def train_one_epoch_bert(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device
) -> Tuple[float, float, float]:
    """训练 BERT 一个 epoch。"""
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []

    for batch in tqdm(loader, desc="BERT Train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    metrics = compute_metrics(y_true, y_pred)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, metrics["accuracy"], metrics["f1"]


@torch.no_grad()
def evaluate_bert(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[float, float, float, np.ndarray, List[int], List[int]]:
    """评估 BERT 微调模型。"""
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    for batch in tqdm(loader, desc="BERT Eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    metrics = compute_metrics(y_true, y_pred)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, metrics["accuracy"], metrics["f1"], metrics["confusion_matrix"], y_true, y_pred


def fit_bert(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    device: torch.device,
    save_path: str
) -> Dict[str, List[float]]:
    """微调 BERT，并执行验证和早停。"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.bert_lr)
    total_steps = len(train_loader) * config.bert_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    early_stopping = EarlyStopping(config.early_stop_patience, save_path, mode="max")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, config.bert_epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch_bert(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc, val_f1, _, _, _ = evaluate_bert(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"[BERT][Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        should_stop, improved = early_stopping.step(val_f1, model)
        if improved:
            print("  验证集 F1 提升，已保存当前最优 BERT 模型。")
        if should_stop:
            print("  触发早停，结束 BERT 微调。")
            break

    model.load_state_dict(torch.load(save_path, map_location=device))
    return history


def train_tfidf_logistic_regression(
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int]
) -> Dict[str, object]:
    """训练 TF-IDF + 逻辑回归基线模型。"""
    vectorizer = TfidfVectorizer(max_features=10000)
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)

    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    clf.fit(x_train, train_labels)
    preds = clf.predict(x_test)

    metrics = compute_metrics(test_labels, preds.tolist())
    return metrics


def describe_dataset(texts: List[str]) -> Dict[str, float]:
    """统计文本长度特征。"""
    lengths = [len(simple_tokenize(text)) for text in texts]
    return {
        "mean_length": float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
        "max_length": float(np.max(lengths)),
        "min_length": float(np.min(lengths))
    }


def main() -> None:
    config = Config()
    set_seed(config.seed)

    os.makedirs(config.save_dir, exist_ok=True)
    model_dir = os.path.join(config.save_dir, "models")
    figure_dir = os.path.join(config.save_dir, "figures")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    training_log_path = setup_training_log(config.save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备：{device}")

    dataset_name, train_texts_raw, train_labels, val_texts_raw, val_labels, test_texts_raw, test_labels = load_binary_text_dataset(config)
    print(f"实际使用数据集：{dataset_name}")
    print(f"训练集大小：{len(train_texts_raw)}，验证集大小：{len(val_texts_raw)}，测试集大小：{len(test_texts_raw)}")

    # 文本清洗
    train_texts = [clean_text(text) for text in tqdm(train_texts_raw, desc="清洗训练集")]
    val_texts = [clean_text(text) for text in tqdm(val_texts_raw, desc="清洗验证集")]
    test_texts = [clean_text(text) for text in tqdm(test_texts_raw, desc="清洗测试集")]

    print("\n===== 预处理示例 =====")
    print("原始文本：", train_texts_raw[0][:250])
    print("清洗后文本：", train_texts[0][:250])
    print("分词结果：", simple_tokenize(train_texts[0])[:30])

    stats = describe_dataset(train_texts)
    print("\n===== 训练集文本长度统计 =====")
    print(
        f"平均长度：{stats['mean_length']:.2f}，中位数：{stats['median_length']:.2f}，"
        f"最长：{stats['max_length']:.0f}，最短：{stats['min_length']:.0f}"
    )

    # 构建词表
    vocab = build_vocab(train_texts, config.max_vocab_size, config.min_freq)
    with open(os.path.join(config.save_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"词表大小：{len(vocab)}")

    # DataLoader 公共设置
    loader_kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available()
    }

    # ---------------------------
    # 1. 从零实现 Transformer
    # ---------------------------
    scratch_train_ds = ScratchTextDataset(train_texts, train_labels, vocab, config.max_length)
    scratch_val_ds = ScratchTextDataset(val_texts, val_labels, vocab, config.max_length)
    scratch_test_ds = ScratchTextDataset(test_texts, test_labels, vocab, config.max_length)

    scratch_train_loader = DataLoader(scratch_train_ds, batch_size=config.batch_size, shuffle=True, **loader_kwargs)
    scratch_val_loader = DataLoader(scratch_val_ds, batch_size=config.batch_size, shuffle=False, **loader_kwargs)
    scratch_test_loader = DataLoader(scratch_test_ds, batch_size=config.batch_size, shuffle=False, **loader_kwargs)

    scratch_model = TransformerClassifier(
        vocab_size=len(vocab),
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        ff_dim=config.ff_dim,
        max_length=config.max_length,
        dropout=config.dropout,
        num_classes=2,
        pad_idx=vocab["[PAD]"]
    ).to(device)

    scratch_model_path = os.path.join(model_dir, "scratch_transformer_best.pt")
    scratch_history = fit_scratch_transformer(
        scratch_model,
        scratch_train_loader,
        scratch_val_loader,
        config,
        device,
        scratch_model_path
    )

    scratch_criterion = nn.CrossEntropyLoss()
    scratch_test_loss, scratch_test_acc, scratch_test_f1, scratch_cm, _, _ = evaluate_scratch(
        scratch_model,
        scratch_test_loader,
        scratch_criterion,
        device
    )

    plot_training_curves(
        scratch_history,
        os.path.join(figure_dir, "scratch_transformer_training_curves.png"),
        "Scratch Transformer"
    )
    plot_confusion_matrix(
        scratch_cm,
        ["Negative", "Positive"],
        os.path.join(figure_dir, "scratch_transformer_confusion_matrix.png"),
        "Scratch Transformer Confusion Matrix"
    )

    print("\n===== Scratch Transformer 测试集结果 =====")
    print(f"test_loss={scratch_test_loss:.4f}, test_acc={scratch_test_acc:.4f}, test_f1={scratch_test_f1:.4f}")
    print("confusion_matrix:")
    print(scratch_cm)

    # ---------------------------
    # 2. BERT 微调
    # ---------------------------
    print("\n开始加载 BERT 分词器与预训练模型...")
    bert_tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name, use_fast=True)
    bert_model = AutoModelForSequenceClassification.from_pretrained(config.bert_model_name, num_labels=2).to(device)

    bert_train_ds = BertTextDataset(train_texts, train_labels, bert_tokenizer, config.max_length)
    bert_val_ds = BertTextDataset(val_texts, val_labels, bert_tokenizer, config.max_length)
    bert_test_ds = BertTextDataset(test_texts, test_labels, bert_tokenizer, config.max_length)

    bert_train_loader = DataLoader(bert_train_ds, batch_size=config.bert_batch_size, shuffle=True, **loader_kwargs)
    bert_val_loader = DataLoader(bert_val_ds, batch_size=config.bert_batch_size, shuffle=False, **loader_kwargs)
    bert_test_loader = DataLoader(bert_test_ds, batch_size=config.bert_batch_size, shuffle=False, **loader_kwargs)

    bert_model_path = os.path.join(model_dir, "bert_best.pt")
    bert_history = fit_bert(
        bert_model,
        bert_train_loader,
        bert_val_loader,
        config,
        device,
        bert_model_path
    )

    bert_test_loss, bert_test_acc, bert_test_f1, bert_cm, _, _ = evaluate_bert(
        bert_model,
        bert_test_loader,
        device
    )

    plot_training_curves(
        bert_history,
        os.path.join(figure_dir, "bert_training_curves.png"),
        "BERT Fine-tuning"
    )
    plot_confusion_matrix(
        bert_cm,
        ["Negative", "Positive"],
        os.path.join(figure_dir, "bert_confusion_matrix.png"),
        "BERT Confusion Matrix"
    )

    print("\n===== BERT 测试集结果 =====")
    print(f"test_loss={bert_test_loss:.4f}, test_acc={bert_test_acc:.4f}, test_f1={bert_test_f1:.4f}")
    print("confusion_matrix:")
    print(bert_cm)

    # ---------------------------
    # 3. TF-IDF + LogisticRegression
    # ---------------------------
    baseline_metrics = train_tfidf_logistic_regression(train_texts, train_labels, test_texts, test_labels)
    plot_confusion_matrix(
        baseline_metrics["confusion_matrix"],
        ["Negative", "Positive"],
        os.path.join(figure_dir, "tfidf_lr_confusion_matrix.png"),
        "TF-IDF + Logistic Regression Confusion Matrix"
    )

    print("\n===== TF-IDF + LogisticRegression 测试集结果 =====")
    print(f"test_acc={baseline_metrics['accuracy']:.4f}, test_f1={baseline_metrics['f1']:.4f}")
    print("confusion_matrix:")
    print(baseline_metrics["confusion_matrix"])

    # ---------------------------
    # 4. 汇总结果并保存
    # ---------------------------
    result_df = pd.DataFrame([
        {"Model": "Scratch Transformer", "Accuracy": scratch_test_acc, "F1": scratch_test_f1},
        {"Model": "BERT Fine-tuning", "Accuracy": bert_test_acc, "F1": bert_test_f1},
        {"Model": "TF-IDF + LogisticRegression", "Accuracy": baseline_metrics["accuracy"], "F1": baseline_metrics["f1"]},
    ])
    result_df.to_csv(os.path.join(config.save_dir, "experiment_results.csv"), index=False, encoding="utf-8-sig")
    print("\n===== 模型性能对比 =====")
    print(result_df)

    summary = {
        "dataset_name": dataset_name,
        "device": str(device),
        "training_log_path": training_log_path,
        "config": config.__dict__,
        "dataset_stats": stats,
        "scratch_transformer": {
            "test_loss": scratch_test_loss,
            "accuracy": scratch_test_acc,
            "f1": scratch_test_f1,
            "confusion_matrix": scratch_cm.tolist()
        },
        "bert": {
            "test_loss": bert_test_loss,
            "accuracy": bert_test_acc,
            "f1": bert_test_f1,
            "confusion_matrix": bert_cm.tolist()
        },
        "tfidf_lr": {
            "accuracy": baseline_metrics["accuracy"],
            "f1": baseline_metrics["f1"],
            "confusion_matrix": baseline_metrics["confusion_matrix"].tolist()
        }
    }

    with open(os.path.join(config.save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n实验完成。结果文件已保存到 outputs 目录。")


if __name__ == "__main__":
    main()
