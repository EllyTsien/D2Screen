from argparse import ArgumentParser
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             accuracy_score, precision_score, recall_score, f1_score)

def plot_precision_recall_curve(y_true, y_scores, plot=True):
    """
    计算并绘制 Precision-Recall 曲线，同时返回精确率、召回率、阈值数组以及 Average Precision (AP) 值。
    
    参数:
      y_true: np.array，真实标签（1 表示活性分子，0 表示非活性分子）
      y_scores: np.array，模型预测分数
      plot: bool, 是否绘制图形（默认为 True）
    
    返回:
      precision: 精确率数组
      recall: 召回率数组
      thresholds: 阈值数组
      ap: Average Precision 值
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.', label=f'Precision-Recall (AP = {ap:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return precision, recall, thresholds, ap

def binarize_scores(y_scores, threshold=0.5):
    """
    根据给定的阈值将连续预测分数转换为二值预测结果（0 或 1）。
    
    参数:
      y_scores: np.array, 模型预测分数
      threshold: float, 阈值（默认为 0.5）
    
    返回:
      y_pred: np.array, 二值预测结果
    """
    y_pred = (y_scores >= threshold).astype(int)
    return y_pred

def compute_classification_metrics(y_true, y_pred):
    """
    计算分类指标：准确率、精确度（Precision）、召回率（Recall）和 F1 分数。
    
    参数:
      y_true: np.array, 真实标签
      y_pred: np.array, 模型转换后的二值预测标签
    
    返回:
      metrics: dict，包含 'accuracy', 'precision', 'recall' 和 'f1' 的指标值
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    return metrics

def compute_enrichment_factor(y_true, y_scores, top_pct=0.01):
    """
    计算 Enrichment Factor (EF)。
    
    参数:
      y_true: np.array，真实标签（1 表示活性，0 表示非活性）
      y_scores: np.array，模型预测分数
      top_pct: float，保留的百分比（例如 0.01 表示前 1%），默认值为 0.01
    
    返回:
      ef: Enrichment Factor 值；如果总体活性分子数量为 0，则返回 None。
    
    计算公式:
      EF = (前 top_pct 内的活性分子比例) / (总体活性分子比例)
    """
    n_total = len(y_true)
    n_actives = np.sum(y_true)
    # 按预测分数降序排列得到索引
    sorted_indices = np.argsort(y_scores)[::-1]
    n_top = max(1, int(np.floor(n_total * top_pct)))
    top_indices = sorted_indices[:n_top]
    n_actives_top = np.sum(y_true[top_indices])
    
    if n_actives == 0:
        return None
    
    ef = (n_actives_top / n_top) / (n_actives / n_total)
    return ef


def compute_bedroc(y_true, y_scores, alpha=20.0):
    """
    计算 bedROC 值，用于评估虚拟筛选中对早期识别活性分子的能力。
    
    参数:
      y_true: np.array，真实标签（1 表示活性，0 表示非活性）
      y_scores: np.array，预测分数
      alpha: float，调节提前识别惩罚程度的参数，常见值有 20.0, 80.5 等
    
    返回:
      bedroc: float，bedROC 值（范围 0~1）
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n = len(y_true)
    n_pos = np.sum(y_true)
    n_neg = n - n_pos

    if n_pos == 0 or n_neg == 0:
        return None

    # 获取排序后的索引（按预测分数降序）
    sorted_indices = np.argsort(y_scores)[::-1]
    ranks = np.arange(1, n + 1)
    y_true_sorted = y_true[sorted_indices]

    # 提取正样本的 rank（位置）
    ri = ranks[y_true_sorted == 1]

    # 计算 BEDROC
    sum_exp = np.sum(np.exp(-alpha * (ri - 1) / n))
    ra = n_pos / n
    bedroc = (sum_exp * alpha) / (n * ra * (1 - np.exp(-alpha))) + (1 - np.exp(-alpha * ra)) / (1 - np.exp(-alpha))
    
    return bedroc


def main(args):
    # passing parameters
    if args.project_name is None:
        print("Using default project name, finetune")
        project_name = "finetune"
    else:
        project_name = args.project_name

    if args.dataset is None:
        finetune_dataset = args.dataset
        print("Using default dataset, input.csv")
        input_ligands_path = 'datasets/input.csv'
        print(f"Using dataset: {input_ligands_path}")
        if not os.path.exists(input_ligands_path):
            raise FileNotFoundError(f"The file '{input_ligands_path}' does not exist.")
        # processed_input_path = 'datasets/train_preprocessed.csv'
    else:
        finetune_dataset = args.dataset
        input_ligands_path = 'datasets/' + args.dataset
        print(f"Using dataset: {input_ligands_path}")
        if not os.path.exists(input_ligands_path):
            raise FileNotFoundError(f"The file '{input_ligands_path}' does not exist.")
        # processed_input_path = 'datasets/train_preprocessed.csv'

    """
    主函数：读取预测和标签文件，根据公共键（例如 ID）合并数据，
    然后计算并输出各项指标。
    
    参数:
      pred_file: str，预测文件路径，要求包含列：ID, SMILES, pred
      label_file: str，标签文件路径，要求包含列：ID, SMILES, label
      score_threshold: float，用于二值化预测分数的阈值，默认 0.5
    """
    # 读取预测数据和标签数据
    target = project_name.split('_')[0]
    df_pred = pd.read_csv(project_name + '/DL_DUDE_result')
    df_label = pd.read_csv('datasets/DUD-E/'+ target +'/test.csv')
    
    # 根据公共键 "ID"（或其它，如 "SMILES"）合并数据
    df_merged = pd.merge(df_pred, df_label, on='ID', suffixes=('_pred', '_label'))
    if df_merged.empty:
        raise ValueError("合并后的数据为空，请检查两个文件的公共键是否一致。")
    
    # 从合并后的数据中提取预测分数和真实标签
    y_scores = df_merged['pred'].values
    y_true = df_merged['label'].values

    print("==== 分类指标和 PR 曲线 ====")
    # 绘制 Precision-Recall 曲线并计算 Average Precision (AP)
    _, _, _, ap = plot_precision_recall_curve(y_true, y_scores, plot=True)
    print("Average Precision (AP):", ap)
    
    # 将预测分数二值化
    y_pred = binarize_scores(y_scores, threshold=args.threshold)
    
    # 计算分类指标
    metrics = compute_classification_metrics(y_true, y_pred)
    print("Accuracy:", metrics['accuracy'])
    print("Precision:", metrics['precision'])
    print("Recall:", metrics['recall'])
    print("F1 Score:", metrics['f1'])

     # 计算 bedROC
    bedroc_score = compute_bedroc(y_true, y_scores, alpha=20.0)
    print("bedROC (alpha=20):", bedroc_score)
    
    # 计算 Enrichment Factor (EF)
    ef_1 = compute_enrichment_factor(y_true, y_scores, top_pct=0.01)
    ef_5 = compute_enrichment_factor(y_true, y_scores, top_pct=0.05)
    print("Enrichment Factor (Top 1%):", ef_1)
    print("Enrichment Factor (Top 5%):", ef_5)
    
    # 保存评估指标到 CSV 文件
    eval_output_path = os.path.join(project_name, "DL_DUDE_evaluation.csv")
    eval_data = {
        "Average Precision (AP)": [ap],
        "Accuracy": [metrics['accuracy']],
        "Precision": [metrics['precision']],
        "Recall": [metrics['recall']],
        "F1 Score": [metrics['f1']],
        "EF@1%": [ef_1],
        "EF@5%": [ef_5],
        "bedROC (α=20)": [bedroc_score]
    }
    df_eval = pd.DataFrame(eval_data)
    df_eval.to_csv(eval_output_path, index=False)
    print(f"保存评估指标到: {eval_output_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--project_name', default='test', type=str, help='Name your project on the wandb website')
    parser.add_argument('--dataset', default='input.csv', type=str, help='Choose dataset (required)')
    parser.add_argument('--n_samples', default=-1, type=int, help='Number of samples (default: all)')
    parser.add_argument('--threshold', default=0.9, type=float, help="Threshold for predict value (defalt 0.9)")
    parser.add_argument('--thread_num', default=1, type=int, help='Number of thread used for finetuning (default: 1)')
    args = parser.parse_args()
    main(args)
