import os
import time
from multiprocessing import Pool
# change
import multiprocessing as mp
from argparse import ArgumentParser
import pandas as pd
import os

from finetune import run_finetune, test, select_best_model
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

    
    #first-stage screen of ZINC20 library
    for index in range(1, 23): 
        test(model_version='1', project_name=project_name, index=index)
    # Sort, filter and log the final result
    sort_and_filter_csv('datasets/DL_pred/result_' + project_name + '.csv', args.threshold, 'datasets/DL_pred/top_' + project_name + '.csv')


if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = ArgumentParser()
    parser.add_argument('--project_name', default='finetune', type=str, help='Name your project on the wandb website')
    parser.add_argument('--dataset', default='input.csv', type=str, help='Choose dataset (required)')
    parser.add_argument('--n_samples', default=-1, type=int, help='Number of samples (default: all)')
    parser.add_argument('--threshold', default=0.9, type=float, help="Threshold for predict value (defalt 0.9)")
    parser.add_argument('--thread_num', default=1, type=int, help='Number of thread used for finetuning (default: 1)')
    args = parser.parse_args()

    main(args)