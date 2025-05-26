import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--project_name', default='finetune', type=str, help='Name your project on the wandb website')
parser.add_argument('--dataset', default='input.csv', type=str, help='Choose dataset (required)')
parser.add_argument('--n_samples', default=None, type=int, help='Number of total samples (default: None)')
parser.add_argument('--ratio', default=None, type=float, help="Ratio of inactive to active samples (default: None)")
parser.add_argument('--thread_num', default=1, type=int, help='Number of thread used for finetuning (default: 1)')
args = parser.parse_args()

if args.ratio is not None:
    assert args.ratio > 0, "ratio should be larger than 0"
if args.n_samples != None:
    assert args.n_samples > 0, "n_samples should be larger than 0"
if args.n_samples is not None and args.n_samples > 0:
    print(f"Extracting {args.n_samples} samples from the dataset")

# 读取数据集        
df = pd.read_csv(args.dataset)


# 分离 active 和 inactive 数据
df_active = df[df['label'] == 1]
df_inactive = df[df['label'] == 0]

# 查看数量
n_active = len(df_active)
print(f"Active count: {n_active}")
n_inactive = len(df_inactive)
print(f"Inactive count: {n_inactive}")
print(f'Ratio of active to inactive: {n_inactive / n_active:.2f}')
# 如果 ratio 没有指定，则不进行采样
if args.ratio is None:
    # 直接退出
    exit(0)
# 如果只指定 ratio，则根据active比例采样
if args.ratio is not None and args.n_samples is None:
    target_inactive = int(min(n_inactive, n_active * args.ratio))
    df_inactive_sampled = df_inactive.sample(n=target_inactive, random_state=42)
    df_balanced = pd.concat([df_active, df_inactive_sampled]).sample(frac=1, random_state=42)
    save_path = args.dataset.split('.')[0] + f'_ratio_{args.ratio}.csv'
    df_balanced.to_csv(save_path, index=False)
    print(f"Saved to {save_path}")
    exit(0)

# 如果同时指定 ratio 和 n_samples
if args.ratio is not None and args.n_samples is not None:
    # 设定比例和总样本数计算 active / inactive 数量
    active_count = int(args.n_samples / (1 + args.ratio))
    inactive_count = args.n_samples - active_count

    # 取交集，防止样本不够用
    if active_count > n_active:
        print(f"Active count {active_count} exceeds available {n_active}, using all active samples")
        exit(0)
    if inactive_count > n_inactive:     
        print(f"Inactive count {inactive_count} exceeds available {n_inactive}, using all inactive samples")
        exit(0)

    df_active_sampled = df_active.sample(n=active_count, random_state=42)
    df_inactive_sampled = df_inactive.sample(n=inactive_count, random_state=42)

    df_balanced = pd.concat([df_active_sampled, df_inactive_sampled]).sample(frac=1, random_state=42)
    save_path = args.dataset.split('.')[0] + f'_ratio_{args.ratio}_nsamples_{args.n_samples}.csv'
    df_balanced.to_csv(save_path, index=False)
    print(f"Saved to {save_path}")
    exit(0)

# 如果只指定 n_samples 而未指定 ratio，则按原始分布比例抽样
if args.ratio is None and args.n_samples is not None:
    df_sampled = df.sample(n=min(args.n_samples, len(df)), random_state=42)
    save_path = args.dataset.split('.')[0] + f'_nsamples_{args.n_samples}.csv'
    df_sampled.to_csv(save_path, index=False)
    print(f"Saved to {save_path}")


