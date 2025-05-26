import multiprocessing as mp
from argparse import ArgumentParser
import os

from finetune import test, select_best_model
from dataloader import sort_and_filter_csv

model_version = '1'

def main(args):
    # passing parameters
    if args.project_name is None:
        print("Using default project name, finetune")
        project_name = "finetune"
    else:
        project_name = args.project_name
    
    if args.threshold is None:
        print("Using default threshold, 0.9")
        args.threshold = 0.9

    
    select_best_model(model_version, project_name)
    
    #first-stage screen of ZINC20 library
    for index in range(1, 10): 
        test(model_version='1', project_name=project_name, index=index)
    # Sort, filter and log the final result
    sort_and_filter_csv(project_name + '/ZINC20/ZINC20_DL_result.csv', args.threshold, project_name + '/ZINC20/DL_top.csv')


if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = ArgumentParser()
    parser.add_argument('--project_name', default='finetune', type=str, help='Name your project on the wandb website')
    parser.add_argument('--n_samples', default=-1, type=int, help='Number of samples (default: all)')
    parser.add_argument('--threshold', default=0.9, type=float, help="Threshold for predict value (defalt 0.9)")
    parser.add_argument('--thread_num', default=1, type=int, help='Number of thread used for finetuning (default: 1)')
    args = parser.parse_args()

    main(args)