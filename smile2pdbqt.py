import pandas as pd
from argparse import ArgumentParser
from ligand_prep import LigandPrep
import multiprocessing as mp
import os


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
    else:
        finetune_dataset = args.dataset
        input_ligands_path = args.dataset
        print(f"Using dataset: {input_ligands_path}")
        if not os.path.exists(input_ligands_path):
            raise FileNotFoundError(f"The file '{input_ligands_path}' does not exist.")
        
    if args.grid_center is None:
        print("Using default grid center, datasets/crystal_ligand.pdb")
        grid_center = 'datasets/crystal_ligand.pdb'
        if not os.path.exists(grid_center):
            raise FileNotFoundError(f"The file '{grid_center}' does not exist.")
    else:
        grid_center = args.grid_center
        print(f"Using grid center: {grid_center}")
        if not os.path.exists(grid_center):
            raise FileNotFoundError(f"The file '{grid_center}' does not exist.")
    
    # read top.csv
    output_filefolder = os.path.dirname(input_ligands_path.rsplit(".", 1)[0]+f"/ligand_prep/")
    if not os.path.exists(output_filefolder):
        os.makedirs(output_filefolder)
        print(f"创建文件夹: {output_filefolder}")

    df = pd.read_csv(input_ligands_path)
    for index, row in df.iterrows():
        smiles = row['SMILES']
        ligand_id = row['ID']
        output_file = input_ligands_path.rsplit(".", 1)[0]+f"/ligand_prep/ligand_{ligand_id}.sdf"

        try:
            ligand_prep = LigandPrep(smiles,grid_center)
            ligand_prep.smile2sdf(output_file)
            print(f"分子已保存为 {output_file}")
        except ValueError as e:
            print(e)
            print(f"行 {index} 的 SMILES 无效: {smiles}")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = ArgumentParser()
    parser.add_argument('--project_name', default='finetune', type=str, help='Name your project on the wandb website')
    parser.add_argument('--dataset', default='datasets/DL_pred/top.csv', type=str, help='Choose dataset (required)')
    parser.add_argument('--grid_center', default='datasets/crystal_ligand.pdb', type=str, help='Specify grid center (required)')
    parser.add_argument('--n_samples', default=-1, type=int, help='Number of samples (default: all)')
    parser.add_argument('--threshold', default=0.9, type=float, help="Threshold for predict value (defalt 0.9)")
    parser.add_argument('--thread_num', default=1, type=int, help='Number of thread used for finetuning (default: 1)')
    args = parser.parse_args()

    main(args)