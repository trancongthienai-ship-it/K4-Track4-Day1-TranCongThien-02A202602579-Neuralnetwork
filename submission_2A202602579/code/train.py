"""train.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import time
import copy
import random
import csv

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # TODO: chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    
    p_denom = tp + fp
    r_denom = tp + fn
    
    precision = np.zeros_like(tp, dtype=float)
    np.divide(tp, p_denom, out=precision, where=p_denom != 0)
    
    recall = np.zeros_like(tp, dtype=float)
    np.divide(tp, r_denom, out=recall, where=r_denom != 0)
    
    f1_denom = precision + recall
    f1 = np.zeros_like(tp, dtype=float)
    np.divide(2 * precision * recall, f1_denom, out=f1, where=f1_denom != 0)
    
    return float(np.mean(f1))


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i+batch_size]
        logits = model(xb)
        preds.append(logits.argmax(dim=1))
    return torch.cat(preds)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    
    for i in range(0, len(X), batch_size):
        xb = X[i:i+batch_size]
        yb = y[i:i+batch_size]
        logits = model(xb)
        
        if loss_name == "ce":
            batch_loss = F.cross_entropy(logits, yb, reduction="sum")
        elif loss_name == "mse":
            y_onehot = F.one_hot(yb, num_classes=logits.shape[-1]).float()
            batch_loss = F.mse_loss(logits, y_onehot, reduction="none").mean(dim=1).sum()
        
        total_loss += batch_loss.item()
        all_preds.append(logits.argmax(dim=1))
        
    all_preds = torch.cat(all_preds)
    mean_loss = total_loss / len(X)
    acc = (all_preds == y).float().mean().item()
    
    cm = np.zeros((7, 7), dtype=int)
    y_np = y.cpu().numpy()
    p_np = all_preds.cpu().numpy()
    np.add.at(cm, (y_np, p_np), 1)
    
    macro_f1 = macro_f1_from_confusion(cm)
    
    return {"loss": mean_loss, "acc": acc, "macro_f1": macro_f1}


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    elif loss_name == "mse":
        y_onehot = F.one_hot(y, num_classes=logits.shape[-1]).float()
        return F.mse_loss(logits, y_onehot)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    (tên khoá của summary trùng tên cột trong experiments.xlsx)

    Các bước:
      0. set_seed(cfg["seed"]); tạo model = MLP(...), assert count_params(model) == EXPECTED_PARAMS[hidden]
         chuyển model lên device; tạo optimizer = build_optimizer(...)
         nếu precision == "fp16": scaler = torch.amp.GradScaler(...)
      1. step0_loss = evaluate(model, X_val, y_val)["loss"]   # TRƯỚC bước cập nhật đầu tiên; kỳ vọng ≈ ln 7
      2. for epoch in 1..epochs:
           model.train()
           for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
               with torch.autocast(...)  nếu precision != "fp32":   # chỉ bọc forward + loss
                   logits = model(xb); loss = compute_loss(logits, yb, cfg["loss"])
               optimizer.zero_grad(set_to_none=True)
               backward (qua scaler nếu fp16)
               nếu fp16 và có clip: scaler.unscale_(optimizer)  TRƯỚC khi clip
               gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt; ghi lại
               bước cập nhật (scaler.step(optimizer); scaler.update() nếu fp16, ngược lại optimizer.step())
               nếu loss là NaN/inf: đặt diverged=True và dừng sớm, ĐỪNG để notebook treo
           cuối epoch (dùng evaluate, chế độ eval):
               train_loss trên toàn bộ train (hoặc 1 tập con CỐ ĐỊNH ~50 000 mẫu), val_loss/val_acc/val_macro_f1
               grad_norm trung bình của epoch; thời gian epoch (torch.cuda.synchronize() nếu dùng GPU)
               nếu val_loss tốt nhất từ trước tới giờ: lưu best_state (bản sao state_dict) và best_epoch
      3. tổng hợp summary tại best_epoch (val_acc, val_macro_f1 lấy ở best_epoch); peak_mem_MB nếu có GPU
    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.
    """
    set_seed(cfg["seed"])
    hidden = cfg["hidden"]
    model = MLP(hidden=hidden, dropout=cfg["dropout"], init=cfg["init"], 
                in_features=data["X_tr"].shape[1], num_classes=7)
    
    expected_params = EXPECTED_PARAMS.get(hidden)
    if expected_params:
        assert count_params(model) == expected_params, f"Params = {count_params(model)}, Expected = {expected_params}"
    
    device = data["X_tr"].device
    model.to(device)
    
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"], 
                                weight_decay=cfg["weight_decay"], momentum=cfg["momentum"])
    
    precision = cfg.get("precision", "fp32")
    # For newer PyTorch versions, it's recommended to use torch.amp.GradScaler('cuda')
    # but torch.cuda.amp.GradScaler() is universally supported in older versions
    scaler = torch.amp.GradScaler('cuda') if precision == "fp16" and device.type == 'cuda' else None
    
    step0_res = evaluate(model, data["X_val"], data["y_val"], cfg["loss"])
    step0_loss = step0_res["loss"]
    
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [],
               "val_macro_f1": [], "grad_norm": [], "epoch_time_s": []}
    
    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    diverged = False
    
    epochs = cfg["epochs"]
    batch_size = cfg["batch"]
    
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg["seed"])
    
    # Pre-select subset of train data for faster eval
    eval_tr_size = min(50000, len(data["X_tr"]))
    eval_X_tr = data["X_tr"][:eval_tr_size]
    eval_y_tr = data["y_tr"][:eval_tr_size]
    
    for epoch in range(1, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        
        model.train()
        epoch_grad_norms = []
        
        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size, generator):
            if precision == "fp16" and device.type == 'cuda':
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])
            elif precision == "bf16" and device.type == 'cuda':
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])
                
            optimizer.zero_grad(set_to_none=True)
            
            if scaler:
                scaler.scale(loss).backward()
                if cfg.get("clip_norm") is not None:
                    scaler.unscale_(optimizer)
            else:
                loss.backward()
                
            gn = clip_gradients(model.parameters(), cfg.get("clip_norm"))
            epoch_grad_norms.append(gn)
            
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                break
                
        if diverged:
            print(f"Epoch {epoch}: Diverged!")
            break
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_time = time.time() - start_time
        
        train_res = evaluate(model, eval_X_tr, eval_y_tr, cfg["loss"])
        val_res = evaluate(model, data["X_val"], data["y_val"], cfg["loss"])
        
        history["epoch"].append(epoch)
        history["train_loss"].append(train_res["loss"])
        history["val_loss"].append(val_res["loss"])
        history["val_acc"].append(val_res["acc"])
        history["val_macro_f1"].append(val_res["macro_f1"])
        history["grad_norm"].append(float(np.mean(epoch_grad_norms)))
        history["epoch_time_s"].append(epoch_time)
        
        if val_res["loss"] < best_val_loss:
            best_val_loss = val_res["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            
    if best_state is None and not diverged:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epochs
        
    peak_mem = 0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        
    summary = {
        "step0_loss": step0_loss,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else float('nan'),
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else float('nan'),
        "val_acc": history["val_acc"][best_epoch-1] if best_epoch > 0 else float('nan'),
        "val_macro_f1": history["val_macro_f1"][best_epoch-1] if best_epoch > 0 else float('nan'),
        "time_per_epoch_s": np.mean(history["epoch_time_s"]) if history["epoch_time_s"] else 0.0,
        "peak_mem_MB": peak_mem,
        "diverged": diverged
    }
    
    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state
    }


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    with open(path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "pred"])
        for r, p in zip(row_id, preds):
            writer.writerow([r, p])


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    hidden = cfg["hidden"]
    model = MLP(hidden=hidden, dropout=cfg["dropout"], init=cfg["init"], 
                in_features=data["X_eval"].shape[1], num_classes=7)
    model.load_state_dict(result["best_state"])
    device = data["X_eval"].device
    model.to(device)
    
    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    print(f"Lưu file dự đoán thành công tại: {pred_path}")
    print(f"Để chấm điểm, hãy chạy lệnh: python scripts/evaluate.py --pred {pred_path} --out eval_result.json")
