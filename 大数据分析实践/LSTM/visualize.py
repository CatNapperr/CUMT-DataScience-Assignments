import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from data_preprocessing import load_and_preprocess, LABEL_NAMES
from models import LSTMAttention

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "outputs"

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})


def plot_loss_curves(histories):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, h) in zip(axes, histories.items()):
        ax.plot(h["train_loss"], label="Train Loss", linewidth=1.5)
        ax.plot(h["val_loss"], label="Val Loss", linewidth=1.5)
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/loss_curve.png")
    plt.close(fig)
    print(f"Saved {OUTPUT_DIR}/loss_curve.png")


def plot_accuracy_curves(histories):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    for (name, h), c in zip(histories.items(), colors):
        ax.plot(h["val_acc"], label=name, linewidth=2, color=c)
    ax.set_title("Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/accuracy_curve.png")
    plt.close(fig)
    print(f"Saved {OUTPUT_DIR}/accuracy_curve.png")


def plot_confusion_matrix(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x)
        all_preds.extend(logits.argmax(-1).cpu().tolist())
        all_labels.extend(y.tolist())

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    ax.set_title("Confusion Matrix — LSTM+Attention (Test Set)")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
    plt.close(fig)
    print(f"Saved {OUTPUT_DIR}/confusion_matrix.png")

    # Also print per-class metrics
    print("\nPer-class metrics:")
    for i, name in enumerate(LABEL_NAMES):
        tp = cm[i, i]
        total_pred = cm[:, i].sum()
        total_true = cm[i, :].sum()
        prec = tp / total_pred if total_pred > 0 else 0
        rec  = tp / total_true if total_true > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"  {name:<18} Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}  Support={total_true}")


def plot_attention_heatmap(model, loader, num_samples=8):
    model.eval()
    x_batch, y_batch = next(iter(loader))
    x_batch = x_batch[:num_samples].to(DEVICE)

    _, attn_weights = model(x_batch, return_attention=True)
    attn_weights = attn_weights.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(attn_weights.T, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(num_samples))
    ax.set_yticks(range(3))
    ax.set_yticklabels(["Season -2", "Season -1", "Current Season"])
    ax.set_xlabel("Sample Index")
    ax.set_title("Attention Weights Over 3 Historical Seasons")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Weight")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/attention_heatmap.png")
    plt.close(fig)

    # Average weight distribution
    avg_weights = attn_weights.mean(axis=0)
    print(f"\nAverage attention weight per season position:")
    for i, label in enumerate(["Season -2", "Season -1", "Current Season"]):
        print(f"  {label}: {avg_weights[i]:.4f}")
    print(f"Saved {OUTPUT_DIR}/attention_heatmap.png")


def main():
    (X_train, y_train), (X_val, y_val), (X_test, y_test), _, _, _ = load_and_preprocess(
        "data/NBA_Season_Stats.csv"
    )

    # Load training history
    histories = np.load(f"{OUTPUT_DIR}/training_history.npz", allow_pickle=True)

    plot_loss_curves({
        "MLP": histories["MLP"].item(),
        "LSTM": histories["LSTM"].item(),
        "LSTM_Attention": histories["LSTM_Attention"].item(),
    })
    plot_accuracy_curves({
        "MLP": histories["MLP"].item(),
        "LSTM": histories["LSTM"].item(),
        "LSTM_Attention": histories["LSTM_Attention"].item(),
    })

    # Load best model for confusion matrix + attention
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    test_loader = DataLoader(test_ds, batch_size=64)

    model = LSTMAttention().to(DEVICE)
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/LSTM_Attention.pth", map_location=DEVICE))

    plot_confusion_matrix(model, test_loader)
    plot_attention_heatmap(model, test_loader)


if __name__ == "__main__":
    main()
