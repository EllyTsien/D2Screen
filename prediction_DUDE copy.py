import os
import time
from multiprocessing import Pool
import multiprocessing as mp
from argparse import ArgumentParser
import pandas as pd
from dataloader import sort_and_filter_csv
from finetune import test_DUDE, select_best_model
from preprocess import Input_ligand_preprocess,  SMILES_Transfer

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
        # processed_input_path = 'datasets/train_preprocessed.csv'
    else:
        finetune_dataset = args.dataset
        input_ligands_path = 'datasets/' + args.dataset
        print(f"Using dataset: {input_ligands_path}")
        if not os.path.exists(input_ligands_path):
            raise FileNotFoundError(f"The file '{input_ligands_path}' does not exist.")
        # processed_input_path = 'datasets/train_preprocessed.csv'

    select_best_model(model_version, project_name)
    
    #preprocess DUDE/test_nolabel.csv to .pkl
    processor = Input_ligand_preprocess(input_ligands_path, project_name, 'DUDE')
    processed_input_path = processor.preprocess() 
    processed_input_csv = pd.read_csv(processed_input_path)
    SMILES_transfer = SMILES_Transfer(processed_input_csv, project_name, 'DUDE')
    SMILES_transfer.run()
    print(processed_input_path)
    test_DUDE(model_version='1', project_name=project_name, nolabel_file_path= processed_input_path, index=0)
    # Sort, filter and log the final result
    sort_and_filter_csv(project_name + '/DL_DUDE_result.csv', args.threshold, project_name + '/DL_DUDE_top.csv')


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