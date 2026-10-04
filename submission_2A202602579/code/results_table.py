"""results_table.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Nhiệm vụ: lưu kết quả từng lần chạy ra JSON, rồi điền vào experiments.xlsx từ mẫu
templates/experiment_table_template.xlsx (đừng gõ tay hàng chục dòng, rất dễ sai).

Tên cột của sheet "Experiments" (giữ nguyên, đúng thứ tự mẫu):
    exp_id, group, description, loss, optimizer, lr, weight_decay, batch, epochs, hidden, dropout,
    clip_norm, precision, init, seed, step0_loss, best_val_loss, best_epoch, final_train_loss,
    final_val_loss, val_acc, val_macro_f1, time_per_epoch_s, peak_mem_MB, diverged,
    eval_acc, eval_macro_f1, figure_file, notes
(các cột công thức ở cuối bảng mẫu tự tính, đừng ghi đè)
"""
from __future__ import annotations

import json
from pathlib import Path


def save_result(result: dict, results_dir: str = "../results") -> str:
    """Ghi result["cfg"], result["history"], result["summary"] (KHÔNG ghi best_state) ra
    <results_dir>/<exp_id>.json. Trả về đường dẫn file. Tạo thư mục nếu chưa có."""
    path_dir = Path(results_dir)
    path_dir.mkdir(parents=True, exist_ok=True)
    exp_id = result["cfg"]["exp_id"]
    out_path = path_dir / f"{exp_id}.json"
    
    to_save = {
        "cfg": result["cfg"],
        "history": result["history"],
        "summary": result["summary"]
    }
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_save, f, indent=2)
    return str(out_path)


def load_results(results_dir: str = "../results") -> list[dict]:
    """Đọc mọi file *.json trong results_dir, trả về danh sách dict (sắp theo exp_id)."""
    path_dir = Path(results_dir)
    if not path_dir.exists():
        return []
        
    results = []
    for fpath in sorted(path_dir.glob("*.json")):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            results.append(data)
    return results


def to_row(result: dict, eval_scores: dict | None = None, notes: str = "") -> dict:
    """Biến một kết quả thành một dòng của bảng: gộp cfg + summary (+ eval_acc, eval_macro_f1 nếu có)
    + figure_file = f"figures/{exp_id}.png". Khoá phải trùng tên cột ở đầu file.
    Chỉ truyền eval_scores cho baseline và cấu hình cuối cùng."""
    row = {}
    row.update(result["cfg"])
    row.update(result["summary"])
    
    if "hidden" in row and isinstance(row["hidden"], list):
        row["hidden"] = str(tuple(row["hidden"]))
        
    row["figure_file"] = f"{row['exp_id']}.png"
    row["notes"] = notes
    
    if eval_scores:
        row["eval_acc"] = eval_scores.get("acc")
        row["eval_macro_f1"] = eval_scores.get("macro_f1")
        
    return row


def write_xlsx(rows: list[dict], template_path: str, out_path: str) -> None:
    """Điền các dòng vào sheet "Experiments" của mẫu, từ dòng 2 trở xuống, rồi lưu thành out_path.

    Các bước (openpyxl):
      1. wb = openpyxl.load_workbook(template_path)   # KHÔNG dùng data_only=True (sẽ mất công thức)
      2. ws = wb["Experiments"]; đọc tiêu đề dòng 1 để biết cột nào ứng với khoá nào
      3. với mỗi row: ghi giá trị vào đúng cột; BỎ QUA các cột công thức (step0_gap_vs_lnC, gap_val_minus_train,
         delta_val_f1_vs_base, beyond_noise)
      4. wb.save(out_path)
    Sau khi lưu, mở file bằng Excel/LibreOffice để các công thức tính lại.
    """
    import openpyxl
    wb = openpyxl.load_workbook(template_path)
    ws = wb["Experiments"]
    
    headers = [cell.value for cell in ws[1]]
    start_row = 2
    
    for row_dict in rows:
        for col_idx, header in enumerate(headers, 1):
            if header in row_dict:
                val = row_dict[header]
                if isinstance(val, (list, tuple)):
                    val = str(val)
                ws.cell(row=start_row, column=col_idx, value=val)
        start_row += 1
        
    wb.save(out_path)
