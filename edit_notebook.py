import json

nb_path = 'code/lab.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 1: Environment
nb['cells'][1]['source'] = [
    "# ===== Cấu hình đường dẫn và môi trường =====\n",
    "import os, sys, json, time, subprocess\n",
    "import numpy as np, torch\n",
    "\n",
    "REPO_ROOT = \"../..\"     # thư mục gốc repo (chứa data/ và scripts/), tính từ submission_<MSSV>/code/\n",
    "OUT_DIR   = \"..\"        # nơi ghi figures/, results/, experiments.xlsx, predictions_eval.csv\n",
    "\n",
    "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "print(\"PyTorch version:\", torch.__version__)\n",
    "if device == \"cuda\":\n",
    "    print(\"GPU:\", torch.cuda.get_device_name(0))\n",
    "\n",
    "os.makedirs(f\"{OUT_DIR}/figures\", exist_ok=True)\n",
    "os.makedirs(f\"{OUT_DIR}/results\", exist_ok=True)\n",
    "\n",
    "from data import prepare_data\n",
    "from model import MLP, EXPECTED_PARAMS, count_params, init_weights, activation_stats\n",
    "from optimizer import build_optimizer, clip_gradients\n",
    "from train import DEFAULT_CFG, set_seed, evaluate, predict, run_experiment, final_eval\n",
    "from plots import plot_run, plot_compare\n",
    "from results_table import save_result, load_results, to_row, write_xlsx\n"
]

# Cell 3: Data
nb['cells'][3]['source'] = [
    "# Chạy scripts/split_data.py\n",
    "subprocess.run([sys.executable, \"scripts/split_data.py\"], cwd=REPO_ROOT, check=True)\n",
    "\n",
    "data = prepare_data(device, val_fraction=0.2, seed=42, processed_dir=f\"{REPO_ROOT}/data/processed\")\n",
    "\n",
    "print(f\"X_tr shape: {data['X_tr'].shape}\")\n",
    "print(f\"X_val shape: {data['X_val'].shape}\")\n",
    "print(f\"X_eval shape: {data['X_eval'].shape}\")\n",
    "\n",
    "# Kiểm tra mean, std của 10 cột đầu tập train\n",
    "numeric_mean = data['X_tr'][:, :10].mean(dim=0)\n",
    "numeric_std = data['X_tr'][:, :10].std(dim=0)\n",
    "print(f\"Numeric columns mean: {numeric_mean.abs().max().item():.4f} (expect ~0)\")\n",
    "print(f\"Numeric columns std: {numeric_std.mean().item():.4f} (expect ~1)\")\n"
]

# Cell 5: Model
nb['cells'][5]['source'] = [
    "model = MLP(hidden=(256, 128), dropout=0.0, init=\"he\").to(device)\n",
    "assert count_params(model) == EXPECTED_PARAMS[(256,128)], \"Số tham số không khớp\"\n",
    "print(\"Số tham số:\", count_params(model))\n",
    "\n",
    "dummy_x = torch.randn(8, 54, device=device)\n",
    "logits = model(dummy_x)\n",
    "print(\"Logits shape:\", logits.shape)\n",
    "assert logits.shape == (8, 7), \"Shape logits sai\"\n",
    "\n",
    "# Mất mát bước 0\n",
    "import torch.nn.functional as F\n",
    "with torch.no_grad():\n",
    "    loss_step0 = F.cross_entropy(logits, torch.zeros(8, dtype=torch.long, device=device))\n",
    "print(\"Loss ngẫu nhiên:\", loss_step0.item(), \"Kỳ vọng:\", np.log(7))\n"
]

# Cell 8: Pipeline
nb['cells'][8]['source'] = [
    "# Chạy baseline vài seed để xem đường cong và tính độ nhiễu\n",
    "cfg_baseline = {**DEFAULT_CFG, \"lr\": 0.05, \"epochs\": 20, \"batch\": 512, \"group\": \"baseline\", \"description\": \"Baseline SGD momentum\"}\n",
    "baseline_results = []\n",
    "\n",
    "for s in [1, 2, 3]:\n",
    "    cfg = {**cfg_baseline, \"seed\": s, \"exp_id\": f\"base-s{s}\"}\n",
    "    res = run_experiment(cfg, data)\n",
    "    save_result(res, f\"{OUT_DIR}/results\")\n",
    "    plot_run(res, f\"{OUT_DIR}/figures/{cfg['exp_id']}.png\")\n",
    "    baseline_results.append(res)\n",
    "    print(f\"Seed {s}: Val Macro-F1 = {res['summary']['val_macro_f1']:.4f}, Acc = {res['summary']['val_acc']:.4f}\")\n",
    "\n",
    "val_f1s = [r['summary']['val_macro_f1'] for r in baseline_results]\n",
    "print(f\"\\nTrung bình Macro-F1: {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}\")\n"
]

# Cell 12: Experiment 1
nb['cells'][12]['source'] = [
    "# Thí nghiệm: thay đổi Optimizer sang AdamW\n",
    "cfg_adamw = {**DEFAULT_CFG, \"optimizer\": \"adamw\", \"lr\": 0.001, \"epochs\": 20, \"batch\": 512, \n",
    "             \"exp_id\": \"opt-adamw\", \"group\": \"optimizer\", \"description\": \"AdamW\"}\n",
    "res_adamw = run_experiment(cfg_adamw, data)\n",
    "save_result(res_adamw, f\"{OUT_DIR}/results\")\n",
    "plot_run(res_adamw, f\"{OUT_DIR}/figures/{cfg_adamw['exp_id']}.png\")\n",
    "print(f\"AdamW: Val Macro-F1 = {res_adamw['summary']['val_macro_f1']:.4f}\")\n"
]

# Cell 14: Compare
nb['cells'][14]['source'] = [
    "# Compare optimizer\n",
    "plot_compare([baseline_results[0], res_adamw], \"val_macro_f1\", f\"{OUT_DIR}/figures/compare_optimizer_f1.png\", \"Compare Optimizer F1\")\n",
    "plot_compare([baseline_results[0], res_adamw], \"val_loss\", f\"{OUT_DIR}/figures/compare_optimizer_loss.png\", \"Compare Optimizer Loss\")\n"
]

# Cell 16: Eval
nb['cells'][16]['source'] = [
    "# Dùng cấu hình tốt nhất làm final (VD: AdamW)\n",
    "cfg_final = cfg_adamw\n",
    "final_eval(cfg_final, res_adamw, data, f\"{OUT_DIR}/predictions_eval.csv\")\n",
    "\n",
    "# Chạy chấm điểm\n",
    "subprocess.run([sys.executable, \"scripts/evaluate.py\", \"--pred\", f\"{OUT_DIR}/predictions_eval.csv\", \"--out\", f\"{OUT_DIR}/eval_result.json\"], cwd=REPO_ROOT, check=True)\n",
    "\n",
    "with open(f\"{OUT_DIR}/eval_result.json\") as f:\n",
    "    eval_res = json.load(f)\n",
    "    print(\"Final Eval Macro-F1:\", eval_res[\"macro_f1\"])\n",
    "    print(\"Final Eval Accuracy:\", eval_res[\"accuracy\"])\n"
]

# Cell 17: Error analysis
nb['cells'][17]['source'] = [
    "# Hiển thị class report\n",
    "import pandas as pd\n",
    "print(\"Classification Report:\")\n",
    "df_report = pd.DataFrame(eval_res[\"class_report\"]).T\n",
    "display(df_report)\n",
    "\n",
    "print(\"Confusion Matrix:\")\n",
    "print(np.array(eval_res[\"confusion_matrix\"]))\n"
]

# Cell 19: Export
nb['cells'][19]['source'] = [
    "# Cập nhật kết quả ra file Excel\n",
    "results = load_results(f\"{OUT_DIR}/results\")\n",
    "rows = [to_row(r) for r in results]\n",
    "# Đổ thêm eval metrics cho dòng final\n",
    "for row in rows:\n",
    "    if row[\"exp_id\"] == cfg_final[\"exp_id\"]:\n",
    "        row[\"eval_acc\"] = eval_res[\"accuracy\"]\n",
    "        row[\"eval_macro_f1\"] = eval_res[\"macro_f1\"]\n",
    "\n",
    "write_xlsx(rows, f\"{REPO_ROOT}/templates/experiment_table_template.xlsx\", f\"{OUT_DIR}/experiments.xlsx\")\n",
    "print(f\"Saved to {OUT_DIR}/experiments.xlsx\")\n"
]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
