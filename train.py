import os
import time
from multiprocessing import Pool
# change
import multiprocessing as mp
from argparse import ArgumentParser
import pandas as pd
import os

from finetune import run_finetune, test, select_best_model, is_task_done
from preprocess import Input_ligand_preprocess,  SMILES_Transfer
from dataloader import sort_and_filter_csv

model_version = '1'

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
        processed_input_path = 'datasets/train_preprocessed.csv'
    else:
        finetune_dataset = args.dataset
        input_ligands_path = 'datasets/' + args.dataset
        print(f"Using dataset: {input_ligands_path}")
        if not os.path.exists(input_ligands_path):
            raise FileNotFoundError(f"The file '{input_ligands_path}' does not exist.")
        processed_input_path = 'datasets/train_preprocessed.csv'

    thread_num =  int( args.thread_num )


    # process train dataset
    processor = Input_ligand_preprocess(input_ligands_path, project_name, 'train')
    processed_input_path = processor.preprocess() 
    processed_input_csv = pd.read_csv(processed_input_path)
    SMILES_transfer = SMILES_Transfer(processed_input_csv, project_name, 'train')
    SMILES_transfer.run()

    
    model_config_list = ["mlp4", "mlp6"]
    lrs_list = [
        ("1e-4", "1e-4"),
        ("5e-3", "5e-3"),
        ("1e-3", "1e-3"),
        ("0", "1e-4"),
        ("0", "5e-3"),
        ("0", "1e-3")

    ]
    drop_list = [0.2, 0.5]
    batch_size_list = [64, 256,516,1024]
    
    '''
    model_config_list = ["mlp4"]
    lrs_list = [
        ("1e-3", "1e-3"),
        ("0", "1e-3")

    ]
    drop_list = [0.2]
    batch_size_list = [32,128]
    '''
    
    os.makedirs(os.path.join(project_name, "checkpoints"), exist_ok=True)
    # 创建参数组合
    tasks = []
    for finetune_model_config in model_config_list:
        for lr, head_lr in lrs_list:
            for dropout_rate in drop_list:
                for batch_size in batch_size_list:
                    for ft_time in range(1, 2):
                        if not is_task_done(project_name, finetune_model_config, lr, head_lr, dropout_rate, ft_time, batch_size):
                            tasks.append((finetune_model_config, lr, head_lr, dropout_rate, ft_time, batch_size, project_name, finetune_dataset, model_version))
                        else:
                            print(f"Skipping trained config: {finetune_model_config}, lr={lr}, head_lr={head_lr}, dropout={dropout_rate}, time={ft_time}, batch={batch_size}")
    
    start_time = time.time()  # 记录开始时间
    # 并行处理任务
    with Pool(thread_num) as p:
        p.map(run_finetune, tasks)
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time
    print(f"Total execution time for finetuning: {elapsed_time:.2f} seconds")

    select_best_model(model_version, project_name)

    '''
    #first-stage screen of ZINC20 library
    for index in range(1, 23): 
        test(model_version='1', project_name=project_name, index=index)
    # Sort, filter and log the final result
    sort_and_filter_csv('datasets/DL_pred/result_' + project_name + '.csv', args.threshold, 'datasets/DL_pred/top_' + project_name + '.csv')
    '''

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = ArgumentParser()
    parser.add_argument('--project_name', default='finetune', type=str, help='Name your project on the wandb website')
    parser.add_argument('--dataset', default='input.csv', type=str, help='Choose dataset (required)')
    parser.add_argument('--n_samples', default=-1, type=int, help='Number of samples (default: all)')
    parser.add_argument('--threshold', default=0.9, type=float, help="Threshold for predict value (defalt 0.9)")
    parser.add_argument('--thread_num', default=1, type=int, help='Number of thread used for finetuning (default: 1)')
    args = parser.parse_args()

    main(args)