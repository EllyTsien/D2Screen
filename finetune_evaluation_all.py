import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve
)

def txt_to_csv(txt_path, csv_path):
    df = pd.read_csv(txt_path, sep='\\s+', header=None, names=["ID_raw", "docking_score"])
    df["ID"] = df["ID_raw"].str.replace("ligand_", "", regex=False).astype(int)
    df = df[["ID", "docking_score"]]
    df.to_csv(csv_path, index=False)
    return csv_path

def compute_enrichment_factor(y_true, y_scores, top_pct=0.01):
    y_true = np.asarray(y_true).astype(bool)
    n_total   = y_true.size
    n_actives = y_true.sum()
    if n_actives == 0:
        return np.nan                       # 全库没活性 → 无 EF

    n_top = max(1, int(np.floor(n_total * top_pct)))
    n_actives_top = y_true[:n_top].sum()

    return (n_actives_top / n_top) / (n_actives / n_total)

def compute_bedroc(y_true, y_scores, alpha=20.0):
    y_true = np.asarray(y_true).astype(int)
    n      = len(y_true)
    n_pos  = y_true.sum()
    n_neg  = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan

    ranks = np.arange(1, n + 1)
    ri    = ranks[y_true == 1]          # 直接在当前顺序取正样本名次

    sum_exp = np.exp(-alpha * (ri - 1) / n).sum()
    ra      = n_pos / n
    bedroc  = (sum_exp * alpha) / (n * ra * (1 - np.exp(-alpha))) \
            + (1 - np.exp(-alpha * ra)) / (1 - np.exp(-alpha))
    return bedroc

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
    output_path = os.path.join(output_dir, f'PR_curve_{strategy_name.replace(" ", "_").replace("(", "").replace(")", "").replace(">", "gt")}.png')
    plt.savefig(output_path)
    plt.close()
    print(f"📈 Saved PR curve for {strategy_name} to: {output_path}")

def evaluate_dl_classification(df, threshold=0.9):
    y_true = df["label"].values
    y_pred = (df["pred"].values > threshold).astype(int)
    y_scores = df["pred"].values

    # Classification metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Ranking metrics
    ap = average_precision_score(y_true, y_scores)
    ef0_5 = compute_enrichment_factor(y_true, y_scores, 0.005)
    ef1 = compute_enrichment_factor(y_true, y_scores, 0.01)
    ef5 = compute_enrichment_factor(y_true, y_scores, 0.05)
    bedroc_8 = compute_bedroc(y_true, y_scores, 8.0)
    bedroc_16 = compute_bedroc(y_true, y_scores, 16.1)
    bedroc_20 = compute_bedroc(y_true, y_scores, 20.0)
    bedroc_160 = compute_bedroc(y_true, y_scores,160.9)
    bedroc_321_9 = compute_bedroc(y_true, y_scores, 321.9)

    # Output
    print("\n== DL Classification ==")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"AP:         {ap:.4f}")
    print(f"EF@0.5%:      {ef0_5:.4f}" if ef1 is not None else "EF@0.5%:      N/A")
    print(f"EF@1%:      {ef1:.4f}" if ef1 is not None else "EF@1%:      N/A")
    print(f"EF@5%:      {ef5:.4f}" if ef5 is not None else "EF@5%:      N/A")
    print(f"bedROC(α = 8.0):     {bedroc_8:.4f}" if bedroc_8 is not None else "bedROC(α = 8.0 :     N/A")
    print(f"bedROC(α = 16.1):     {bedroc_16:.4f}" if bedroc_16 is not None else "bedROC(α = 16.1） :     N/A")
    print(f"bedROC(α = 20.0):     {bedroc_20:.4f}" if bedroc_20 is not None else "bedROC(α = 20.0） :     N/A")
    print(f"bedROC(α = 160.9):     {bedroc_160:.4f}" if bedroc_160 is not None else "bedROC(α = 160.9） :     N/A")
    print(f"bedROC(α = 321.9):     {bedroc_321_9:.4f}" if bedroc_321_9 is not None else "bedROC(α = 321.9） :     N/A")

    return {
        "Strategy": f"DL (threshold>{threshold})",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "AP": ap,
        "EF@0.5%": ef0_5,
        "EF@1%": ef1,
        "EF@5%": ef5,
        "bedROC(α = 8.0)": bedroc_8,
        "bedROC(α = 16.1)": bedroc_16,
        "bedROC(α = 20.0)": bedroc_20,
        "bedROC(α = 160.9)": bedroc_160,
        "bedROC(α = 321.9)": bedroc_321_9
    }


def evaluate_ranking(df, score_column, name, top_k_count=None, output_file=None):
    y_scores = df[score_column].values
    y_true = df["label"].values

    # Ranking-based metrics
    ap = average_precision_score(y_true, y_scores)
    ef0_5 = compute_enrichment_factor(y_true, y_scores, 0.005)
    ef1 = compute_enrichment_factor(y_true, y_scores, 0.01)
    ef5 = compute_enrichment_factor(y_true, y_scores, 0.05)
    bedroc_8 = compute_bedroc(y_true, y_scores, 8.0)
    bedroc_16 = compute_bedroc(y_true, y_scores, 16.1)
    bedroc_20 = compute_bedroc(y_true, y_scores, 20.0)
    bedroc_160 = compute_bedroc(y_true, y_scores, 160.9)
    bedroc_321_9 = compute_bedroc(y_true, y_scores, 321.9)

    # Classification by top-k count
    df_sorted = df.sort_values(by=score_column, ascending=True)
    k = min(top_k_count, len(df_sorted))  # 避免 top_k_count 超出长度
    selected_indices = df_sorted.head(k).index

    # 构造 y_pred，按原顺序
    y_pred = pd.Series(0, index=df.index)
    y_pred.loc[selected_indices] = 1
    y_pred = y_pred.values

    # 分类指标（与 y_true 顺序对齐）
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # 打印所有指标
    print(f"\n== {name} Ranking (Top {top_k_count} entries by {score_column}) ==")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"AP:         {ap:.4f}")
    print(f"EF@0.5%:      {ef0_5:.4f}" if ef1 is not None else "EF@0.5%:      N/A")
    print(f"EF@1%:      {ef1:.4f}" if ef1 is not None else "EF@1%:      N/A")
    print(f"EF@5%:      {ef5:.4f}" if ef5 is not None else "EF@5%:      N/A")
    print(f"bedROC(α = 8.0):     {bedroc_8:.4f}" if bedroc_8 is not None else "bedROC(α = 8.0):     N/A")
    print(f"bedROC(α = 16.1):    {bedroc_16:.4f}" if bedroc_16 is not None else "bedROC(α = 16.1):    N/A")
    print(f"bedROC(α = 20.0):    {bedroc_20:.4f}" if bedroc_20 is not None else "bedROC(α = 20.0):    N/A")
    print(f"bedROC(α = 160.9):   {bedroc_160:.4f}" if bedroc_160 is not None else "bedROC(α = 160.9):   N/A")
    print(f"bedROC(α = 321.9):   {bedroc_321_9:.4f}" if bedroc_321_9 is not None else "bedROC(α = 321.9):   N/A")

    # 保存预测标签到文件
    if output_file is not None:
        df_out = df.copy()
        df_out["y_pred"] = 0
        df_out.loc[selected_indices, "y_pred"] = 1
        df_out.sort_values(by=score_column, ascending=True, inplace=True)
        df_out.to_csv(output_file, index=False)
        print(f"\n[✔] Output saved to {output_file}")

    return {
        "Strategy": f"{name} Ranking (Top {top_k_count})",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "AP": ap,
        "EF@0.5%": ef0_5,
        "EF@1%": ef1,
        "EF@5%": ef5,
        "bedROC(α = 8.0)": bedroc_8,
        "bedROC(α = 16.1)": bedroc_16,
        "bedROC(α = 20.0)": bedroc_20,
        "bedROC(α = 160.9)": bedroc_160,
        "bedROC(α = 321.9)": bedroc_321_9
    }

def evaluate_threshold_D2Screen(df, score_column, name, threshold, output_file=None):
    """
    score_column < threshold 判为阳性。其它指标与 evaluate_ranking_D2Screen 相同。
    """
    y_scores = df[score_column].values
    y_true   = df["label"].values

    # ---------- 排名式指标 ----------
    ap        = average_precision_score(y_true, y_scores)
    ef0_5 = compute_enrichment_factor(y_true, y_scores, 0.005)
    ef1       = compute_enrichment_factor(y_true, y_scores, 0.01)
    ef5       = compute_enrichment_factor(y_true, y_scores, 0.05)
    bedroc_8  = compute_bedroc(y_true, y_scores, 8.0)
    bedroc_16 = compute_bedroc(y_true, y_scores, 16.1)
    bedroc_20 = compute_bedroc(y_true, y_scores, 20.0)
    bedroc_160= compute_bedroc(y_true, y_scores, 160.9)
    bedroc_321_9 = compute_bedroc(y_true, y_scores, 321.9)

    # ---------- 阈值分类 ----------
    y_pred = (df[score_column] < threshold).astype(int).values
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n== {name} (< {threshold}) ==")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"AP:         {ap:.4f}")
    print(f"EF@0.5%:      {ef0_5:.4f}" if ef1 is not None else "EF@0.5%:      N/A")
    print(f"EF@1%:      {ef1:.4f}" if ef1 is not None else "EF@1%:      N/A")
    print(f"EF@5%:      {ef5:.4f}" if ef5 is not None else "EF@5%:      N/A")
    print(f"bedROC(α = 8.0):     {bedroc_8:.4f}"  if bedroc_8  is not None else "bedROC(α = 8.0):     N/A")
    print(f"bedROC(α = 16.1):    {bedroc_16:.4f}" if bedroc_16 is not None else "bedROC(α = 16.1):    N/A")
    print(f"bedROC(α = 20.0):    {bedroc_20:.4f}" if bedroc_20 is not None else "bedROC(α = 20.0):    N/A")
    print(f"bedROC(α = 160.9):   {bedroc_160:.4f}"if bedroc_160 is not None else "bedROC(α = 160.9):   N/A")
    print(f"bedROC(α = 321.9):   {bedroc_321_9:.4f}" if bedroc_321_9 is not None else "bedROC(α = 321.9):   N/A")

    # ---------- 保存 CSV ----------
    if output_file is not None:
        df_out = df.copy()
        df_out["y_pred"] = y_pred
        df_out.sort_values(by=score_column, ascending=True, inplace=True)
        df_out.to_csv(output_file, index=False)
        print(f"[✔] Output saved to {output_file}")

    return {
        "Strategy":        f"{name} (< {threshold})",
        "Accuracy":        acc,
        "Precision":       prec,
        "Recall":          rec,
        "F1 Score":        f1,
        "AP":              ap,
        "EF@0.5%":         ef0_5,
        "EF@1%":           ef1,
        "EF@5%":           ef5,
        "bedROC(α = 8.0)":   bedroc_8,
        "bedROC(α = 16.1)":  bedroc_16,
        "bedROC(α = 20.0)":  bedroc_20,
        "bedROC(α = 160.9)": bedroc_160,
        "bedROC(α = 321.9)": bedroc_321_9
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate DL + docking strategies from txt inputs.")
    parser.add_argument("--dl_file", type=str, required=True, help="DL prediction CSV")
    parser.add_argument("--dock_file_all", type=str, required=True, help="Docking result for ALL molecules (.txt)")
    parser.add_argument("--dock_file_dl", type=str, required=True, help="Docking result for DL-filtered molecules (.txt)")
    parser.add_argument("--label_file", type=str, required=True, help="Label file (CSV)")
    parser.add_argument("--dl_threshold", type=float, default=0.9, help="DL threshold")
    parser.add_argument("--top_k_count", type=int, default=None, help="Top K count for ranking evaluation")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output files")
    parser.add_argument("--score_threshold", type=float, help="Docking score threshold; score < T is positive in D2Screen")


    args = parser.parse_args()
    if args.top_k_count is None:
        raise ValueError("Please specify --top_k_count for ranking evaluation.")
    if not os.path.exists(args.dl_file):
        raise FileNotFoundError(f"DL file not found: {args.dl_file}")
    if not os.path.exists(args.dock_file_all):
        raise FileNotFoundError(f"Docking file for ALL molecules not found: {args.dock_file_all}")      
    if not os.path.exists(args.dock_file_dl):
        raise FileNotFoundError(f"Docking file for DL-filtered molecules not found: {args.dock_file_dl}")
    if not os.path.exists(args.label_file): 
        raise FileNotFoundError(f"Label file not found: {args.label_file}")
    if not os.path.exists(args.output_dir):
        print(f"Output directory does not exist, creating: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)

    # Convert docking txt to CSV
    dock_all_csv = txt_to_csv(args.dock_file_all, "converted_all_docking.csv")
    dock_dl_csv = txt_to_csv(args.dock_file_dl, "converted_dl_docking.csv")

    # Load input data
    df_dl = pd.read_csv(args.dl_file)
    df_label = pd.read_csv(args.label_file)
    df_all_dock = pd.read_csv(dock_all_csv)
    df_dl_dock = pd.read_csv(dock_dl_csv)

    # Build final merged table
    df_final = df_label.copy()
    df_final = df_final.merge(df_dl[["ID", "pred"]], on="ID", how="left")
    df_final = df_final.merge(df_all_dock.rename(columns={"docking_score": "docking_all"}), on="ID", how="left")
    df_joint_score = df_dl[df_dl["pred"] > args.dl_threshold][["ID"]].merge(
        df_dl_dock.rename(columns={"docking_score": "D2Screen"}), on="ID", how="left")
    df_final = df_final.merge(df_dl_dock.rename(columns={"docking_score": "D2Screen"}), on="ID", how="left")
    df_final["D2Screen"] = df_final["D2Screen"].fillna(0)
    df_final = df_final[["ID", "SMILES", "pred", "docking_all", "D2Screen", "label"]]
    df_final.sort_values(by="D2Screen", ascending=True, inplace=True)
    df_final.to_csv(os.path.join(args.output_dir, "evaluation_combined_scores.csv"), index=False)
    print(f"\n✅ Final combined score table saved to: {args.output_dir}/evaluation_combined_scores.csv")

    results = []

    # Strategy 1: DL
    df_dl_full = df_dl.merge(df_label, on="ID")
    df_dl_full.sort_values(by="pred", ascending=False, inplace=True)
    results.append(evaluate_dl_classification(df_dl_full, args.dl_threshold))
    ### evaluation ranking: args.top_k_count输入改成输入百分比，
    results.append(evaluate_ranking(df_dl_full, "pred", "Deep Learning", args.top_k_count, output_file=os.path.join(args.output_dir, "evaluation_dl.csv")))
    plot_precision_recall_curve(df_dl_full["label"].values, df_dl_full["pred"].values, "Deep Learning", args.output_dir)

    # Strategy 2: All docking
    df_dock_all = df_all_dock.merge(df_label, on="ID", how="right")
    df_dock_all["docking_score"] = df_dock_all["docking_score"].fillna(0)
    df_dock_all.sort_values(by="docking_score", ascending=True, inplace=True)
    # df_dock_all["-docking_score"] = -df_dock_all["docking_score"]
    # ount,  output_file = os.path.join(args.output_dir, "evaluation_all_docking.csv")))
    if args.score_threshold != 0:
        results.append(
            evaluate_threshold_D2Screen(
                df_dock_all,
                "docking_score",
                "docking_score_threshold",
                threshold=args.score_threshold,
                output_file=os.path.join(args.output_dir, "evaluation_docking_threshold.csv")
            )
        )
        plot_precision_recall_curve(
            df_dock_all["label"].values,
            -df_dock_all["docking_score"].values,
            f"docking_score <{args.score_threshold}",
            args.output_dir
        )
    plot_precision_recall_curve(df_dock_all["label"].values, -df_dock_all["docking_score"].values, "Docking (All)", args.output_dir)

    # Strategy 3: DL filter + docking
    # df_passed = df_dl[df_dl["pred"] > args.dl_threshold]
    df_d2screen = df_dl_dock.merge(df_label, on="ID", how="right")
    df_d2screen = df_d2screen.rename(columns={"docking_score": "D2Screen"})
    df_d2screen["D2Screen"] = df_d2screen["D2Screen"].fillna(0)
    df_d2screen.sort_values(by="D2Screen", ascending=True, inplace=True)
    if not df_d2screen.empty:
        results.append(evaluate_ranking(df_d2screen, "D2Screen", f"D2Screen ({args.dl_threshold})", args.top_k_count, os.path.join(args.output_dir, "evaluation_D2Screen_ranking.csv")))
        plot_precision_recall_curve(df_d2screen["label"].values, -df_d2screen["D2Screen"].values, f"D2Screen ({args.dl_threshold})", args.output_dir)
        if args.score_threshold != 0:
            results.append(
                evaluate_threshold_D2Screen(
                    df_d2screen,
                    "D2Screen",
                    "D2Screen_threshold",
                    threshold=args.score_threshold,
                    output_file=os.path.join(args.output_dir, "evaluation_D2Screen_threshold.csv")
                )
            )
            plot_precision_recall_curve(
                df_d2screen["label"].values,
                -df_d2screen["D2Screen"].values,
                f"D2Screen <{args.score_threshold}",
                args.output_dir
            )
    else:
        print("\n⚠️ No molecules passed DL threshold; skipping joint strategy.")

    # Save summary
    columns = ["Strategy", "Accuracy", "Precision", "Recall", "F1 Score", "AP", "EF@0.5%", "EF@1%", "EF@5%", "bedROC(α = 8.0)", "bedROC(α = 16.1)", "bedROC(α = 20.0)", "bedROC(α = 160.9)", "bedROC(α = 321.9)"]
    df_result = pd.DataFrame(results)[columns]
    df_result.to_csv(os.path.join(args.output_dir, "evaluation_summary.csv"), index=False)
    print("✅ Saved evaluation_summary.csv")

if __name__ == "__main__":
    main()
