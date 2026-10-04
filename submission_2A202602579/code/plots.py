"""plots.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Các bước: fig, axes = plt.subplots(1, 3, figsize=...); plot; set_title/xlabel/legend;
              fig.savefig(path, dpi=..., bbox_inches="tight"); plt.close(fig)
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """
    history = result["history"]
    cfg = result["cfg"]
    exp_id = cfg["exp_id"]
    
    epochs = history["epoch"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # (1) Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    if result["summary"]["best_epoch"] > 0:
        axes[0].axvline(x=result["summary"]["best_epoch"], color='r', linestyle='--', alpha=0.5)
        
    # (2) Accuracy / Macro F1
    axes[1].plot(epochs, history["val_acc"], label="Val Acc")
    axes[1].plot(epochs, history["val_macro_f1"], label="Val Macro-F1")
    axes[1].set_title("Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    if result["summary"]["best_epoch"] > 0:
        axes[1].axvline(x=result["summary"]["best_epoch"], color='r', linestyle='--', alpha=0.5)
        
    # (3) Grad Norm
    axes[2].plot(epochs, history["grad_norm"], label="Grad Norm", color="green")
    axes[2].set_title("Gradient Norm")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    
    plt.suptitle(f"{exp_id}: opt={cfg.get('optimizer')}, lr={cfg.get('lr')}, hidden={cfg.get('hidden')}")
    plt.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for r in results:
        epochs = r["history"]["epoch"]
        if metric in r["history"]:
            ax.plot(epochs, r["history"][metric], label=r["cfg"]["exp_id"])
            
    ax.set_title(title if title else f"Compare {metric}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
