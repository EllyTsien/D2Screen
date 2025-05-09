#!/usr/bin/env python
from vina import Vina
from argparse import ArgumentParser
import numpy as np
import os
from grid_box import GridBox

def main(args):
    np.random.seed(args.seed)
    if args.project_name is None:
        print("Using default project name, finetune")
        project_name = "finetune"
    else:
        project_name = args.project_name

    if args.receptor is None:
        print("No receptor specified. Use '--receptor <receptor_file_name.pdbqt>'")
        exit(-1)

    if args.ligand_dir is None:
        print("No ligand directory specified. Use '--ligand_dir <ligand_directory>'")
        exit(-1)    
    else:
        print(f"Using ligand directory: {args.ligand_dir}")
        ligand_dir = args.ligand_dir

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

    # Initialize Vina object
    v = Vina()
    # Set receptor
    v.set_receptor(rigid_pdbqt_filename=args.receptor)

    
    GB = GridBox(grid_center)
    center, bxsize = GB.autodock_grid()
    print('Center:', center)
    print('Box size:', bxsize)
    v.compute_vina_maps(center=center, box_size=bxsize)
    
    # Path for the score file
    score_file_path = project_name + '/docking_results/vina_score.txt'
    if not os.path.exists(project_name + '/docking_results'):
        os.makedirs(project_name + '/docking_results')

    # Open the output file for scores
    with open(score_file_path, 'a') as score_file:
        # Loop through each ligand file in the directory
        for ligand_file in os.listdir(ligand_dir):
            if ligand_file.endswith('aligned.pdbqt'):
                ligand_path = os.path.join(ligand_dir, ligand_file)
                
                # Remove 'aligned' suffix from the ligand file name
                folder_name = ligand_file.replace('.pdbqt', '').replace('_aligned', '')
                folder_path = os.path.join(project_name, 'docking_results', folder_name)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                
                # Create log file path with ligand name
                log_file_path = os.path.join(folder_path, f'{folder_name}.log')

                try:
                    v.set_ligand_from_file(ligand_path)

                    # Get docking score and optimization results
                    score = v.score()
                    optimization = v.optimize()
                    
                    # Output file path for docking results
                    output_path = os.path.join(folder_path, f'{folder_name}_docking_results.pdbqt')

                    print(f'Docking for {folder_name} start!')
                    
                    # Perform docking
                    v.dock(exhaustiveness=8)
                    print(f'Docking for {folder_name} finished!')

                    # Write poses to file
                    v.write_poses(pdbqt_filename=output_path, n_poses=5, energy_range=3.0, overwrite=True)
                    print(f"Docking results saved to {output_path}")
                    print(f"Log information saved to {log_file_path}\n")

                    # Extract scores from the resulting .pdbqt file
                    with open(output_path, 'r') as result_file:
                        for line in result_file:
                            if "REMARK VINA RESULT" in line:
                                # Extract score (assume score is the fourth field in the line)
                                fields = line.split()
                                if len(fields) >= 4:
                                    score_value = fields[3]
                                    score_file.write(f"{folder_name} {score_value}\n")
                                # break

                except Exception as e:
                    with open(log_file_path, 'w') as log_file:
                        log_file.write(f"Error processing {ligand_file}: {e}\n")
                    print(f"Error processing {ligand_file}: {e}\n")

    # Sort the scores file by average score value in ascending order
    sort_scores(score_file_path)

def sort_scores(file_path):
    # 读取文件内容，并计算每个名称的评分平均值
    scores = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                name, score = parts
                try:
                    score = float(score)
                    if name in scores:
                        scores[name].append(score)
                    else:
                        scores[name] = [score]
                except ValueError:
                    print(f"Invalid score value: {score}")

    # 计算每个名称的评分平均值
    average_scores = [(name, np.mean(score_list)) for name, score_list in scores.items()]

    # 按平均评分升序排序
    average_scores.sort(key=lambda x: x[1])
    
    # 将排序结果写回到文件
    with open(file_path, 'w') as f:
        for name, avg_score in average_scores:
            f.write(f"{name} {avg_score}\n")

if __name__ == "__main__":
    # Argument parsing moved here
    parser = ArgumentParser()
    parser.add_argument('--receptor', default=None, type=str, help='Processed receptor protein pdb file')
    parser.add_argument('--ligand_dir', default=None, type=str, help='Processed ligand folder')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--grid_center', default='datasets/crystal_ligand.pdb', type=str, help='Specify grid center by input a crystal ligand pdb file(required)')
    parser.add_argument('--project_name', default='test', type=str, help='Name your project on the wandb website')
    args = parser.parse_args()
    main(args)
