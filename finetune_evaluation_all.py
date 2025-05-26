import pandas as pd
import numpy as np
import argparse
import subprocess
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, average_precision_score
)
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def txt_to_csv(txt_path, csv_path):
    df = pd.read_csv(txt_path, sep='\\s+', header=None, names=["ID_raw", "docking_score"])
    # 提取 ligand_123 → 123（转为 int）
    df["ID"] = df["ID_raw"].str.replace("ligand_", "").astype(int)
    df = df[["ID", "docking_score"]]
    df.to_csv(csv_path, index=False)
    return csv_path

def compute_enrichment_factor(y_true, y_scores, top_pct=0.01):
    n_total = len(y_true)
    n_actives = np.sum(y_true)
    sorted_indices = np.argsort(y_scores)[::-1]
    n_top = max(1, int(np.floor(n_total * top_pct)))
    top_indices = sorted_indices[:n_top]
    n_actives_top = np.sum(y_true[top_indices])
    if n_actives == 0:
        return None
    return (n_actives_top / n_top) / (n_actives / n_total)

def compute_bedroc(y_true, y_scores, alpha=20.0):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n = len(y_true)
    n_pos = np.sum(y_true)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    sorted_indices = np.argsort(y_scores)[::-1]
    ranks = np.arange(1, n + 1)
    y_true_sorted = y_true[sorted_indices]
    ri = ranks[y_true_sorted == 1]
    sum_exp = np.sum(np.exp(-alpha * (ri - 1) / n))
    ra = n_pos / n
    return (sum_exp * alpha) / (n * ra * (1 - np.exp(-alpha))) + (1 - np.exp(-alpha * ra)) / (1 - np.exp(-alpha))

def evaluate_dl_classification(df, threshold=0.9):
    y_true = df["label"].values
    y_pred = (df["pred"].values > threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("\n== DL Classification ==")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    return {
        "Strategy": f"DL (threshold>{threshold})",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    }

def evaluate_ranking(df, score_column, name=""):
    y_scores = df[score_column].values
    y_true = df["label"].values
    ap = average_precision_score(y_true, y_scores)
    ef1 = compute_enrichment_factor(y_true, y_scores, 0.01)
    ef5 = compute_enrichment_factor(y_true, y_scores, 0.05)
    bedroc = compute_bedroc(y_true, y_scores)
    print(f"\n== {name} Ranking ==")
    print(f"AP:     {ap:.4f}")
    print(f"EF@1%:  {ef1:.4f}")
    print(f"EF@5%:  {ef5:.4f}")
    if bedroc is not None:
        print(f"bedROC: {bedroc:.4f}")
    else:
        print("bedROC: N/A")
        print("⚠️  Skipped bedROC: Only positive or only negative samples present in this subset.")

    return {
        "Strategy": name,
        "AP": ap,
        "EF@1%": ef1,
        "EF@5%": ef5,
        "bedROC": bedroc
    }

def plot_precision_recall_curve(y_true, y_scores, strategy_name, output_dir):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label=f'{strategy_name} (AP = {ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {strategy_name}')
    plt.legend()
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'PR_curve_{strategy_name.replace(" ", "_")}.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved PR curve for {strategy_name} to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate DL + docking strategies from txt inputs.")
    parser.add_argument("--dl_file", type=str, required=True, help="DL prediction CSV")
    parser.add_argument("--dock_file_all", type=str, required=True, help="Docking result for ALL molecules (.txt)")
    parser.add_argument("--dock_file_dl", type=str, required=True, help="Docking result for DL-filtered molecules (.txt)")
    parser.add_argument("--label_file", type=str, required=True, help="Label file (CSV)")
    parser.add_argument("--dl_threshold", type=float, default=0.9, help="DL threshold")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output files")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 静默转换 docking txt 为 csv
    dock_all_csv = txt_to_csv(args.dock_file_all, "converted_all_docking.csv")
    dock_dl_csv = txt_to_csv(args.dock_file_dl, "converted_dl_docking.csv")

    # 加载数据
    df_dl = pd.read_csv(args.dl_file)
    df_label = pd.read_csv(args.label_file)
    df_all_dock = pd.read_csv(dock_all_csv)
    df_dl_dock = pd.read_csv(dock_dl_csv)

    results = []
     # 创建最终合并输出表
    df_final = df_label
    df_final = df_final.merge(df_dl[["ID", "pred"]], on="ID", how="left")
    df_final = df_final.merge(df_all_dock.rename(columns={"docking_score": "docking_all"}), on="ID", how="left")

    df_joint_score = df_dl[df_dl["pred"] > args.dl_threshold][["ID"]].merge(
        df_dl_dock.rename(columns={"docking_score": "D2Screen"}), on="ID", how="left")

    df_final = df_final.merge(df_joint_score, on="ID", how="left")
    df_final["D2Screen"] = df_final["D2Screen"].fillna(0)
    df_final = df_final[["ID", "SMILES", "pred", "docking_all", "D2Screen", "label"]]

    # 保存最终表
    df_final.sort_values(by="D2Screen", ascending=True, inplace=True)
    if os.path.exists(os.path.join(args.output_dir, "evaluation_combined_scores.csv")):
        os.remove(os.path.join(args.output_dir, "evaluation_combined_scores.csv"))
    df_final.to_csv(os.path.join(args.output_dir, "evaluation_combined_scores.csv"), index=False)
    print(f"\n✅ Final combined score table saved to: {args.output_dir}")

    # Strategy 1: DL + label
    df_dl_full = df_dl.merge(df_label, on="ID")
    results.append(evaluate_dl_classification(df_dl_full, args.dl_threshold))
    results.append(evaluate_ranking(df_dl_full, "pred", "Deep Learning"))
    plot_precision_recall_curve(df_dl_full["label"].values, df_dl_full["pred"].values, "Deep Learning", args.output_dir)


    # Strategy 2: All docking + label
    df_dock_all = df_all_dock.merge(df_label, on="ID")
    df_dock_all["-docking_score"] = -df_dock_all["docking_score"]
    results.append(evaluate_ranking(df_dock_all, "-docking_score", "Docking (All)"))
    plot_precision_recall_curve(df_dock_all["label"].values, -df_dock_all["docking_score"].values, "Docking (All)", args.output_dir)


    # Strategy 3: DL > threshold → selected docking + label
    df_passed = df_dl[df_dl["pred"] > args.dl_threshold]
    df_dock_joint = df_dl_dock.merge(df_passed[["ID"]], on="ID").merge(df_label, on="ID")
    if not df_dock_joint.empty:
        df_dock_joint["-docking_score"] = -df_dock_joint["docking_score"]
        results.append(evaluate_ranking(df_dock_joint, "-docking_score", f"D2Screen ({args.dl_threshold})"))
        plot_precision_recall_curve(df_dock_joint["label"].values, -df_dock_joint["docking_score"].values, f"D2Screen ({args.dl_threshold})", args.output_dir)
    else:
        print("\n⚠️ No molecules passed DL threshold; skipping joint strategy.")

    # 保存 summary 指标
    df_result = pd.DataFrame(results)
    df_result.to_csv(os.path.join(args.output_dir, "evaluation_summary.csv"), index=False)
    print("✅ Saved evaluation_summary.csv")

    # 创建最终合并输出表
    df_final = df_label[["ID"]].copy()
    df_final = df_final.merge(df_dl[["ID", "pred"]], on="ID", how="left")
    df_final = df_final.merge(df_all_dock.rename(columns={"docking_score": "docking_all"}), on="ID", how="left")

    df_joint_score = df_dl[df_dl["pred"] > args.dl_threshold][["ID"]].merge(
        df_dl_dock.rename(columns={"docking_score": "D2Screen"}), on="ID", how="left")

    df_final = df_final.merge(df_joint_score, on="ID", how="left")
    df_final["D2Screen"] = df_final["D2Screen"].fillna(0)

    # 保存最终表
    if os.path.exists(os.path.join(args.output_dir, "evaluation_combined_scores.csv")):
        os.remove(os.path.join(args.output_dir, "evaluation_combined_scores.csv"))
    df_final.to_csv(os.path.join(args.output_dir, "evaluation_combined_scores.csv"), index=False)
    print(f"✅ Final combined score table saved to: {args.output_dir}evaluation_combined_scores.csv")

if __name__ == "__main__":
    main()