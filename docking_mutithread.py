import concurrent.futures
import os
from vina import Vina
from grid_box import GridBox
import numpy as np
from argparse import ArgumentParser
import time

def process_ligand(ligand_file, ligand_dir, score_file_path, receptor_file):
    v = Vina()

    # Initialize Vina object
    v.set_receptor(rigid_pdbqt_filename=receptor_file)
    v.compute_vina_maps(center=center, box_size=bxsize)
    
    ligand_path = os.path.join(ligand_dir, ligand_file)
    
    # Remove 'aligned' suffix from the ligand file name
    folder_name = ligand_file.replace('.pdbqt', '').replace('_aligned', '')
    folder_path = os.path.join('docking_results', folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Create log file path with ligand name
    log_file_path = os.path.join(folder_path, f'{folder_name}.log')

    try:
        v.set_ligand_from_file(ligand_path)

        # Perform docking
        v.dock(exhaustiveness=8)
        print(f'Docking for {folder_name} finished!')

        # Output file path for docking results
        output_path = os.path.join(folder_path, f'{folder_name}_docking_results.pdbqt')
        v.write_poses(pdbqt_filename=output_path, n_poses=5, energy_range=3.0, overwrite=True)
        print(f"Docking results saved to {output_path}")

        # Extract scores from the resulting .pdbqt file
        with open(output_path, 'r') as result_file:
            for line in result_file:
                if "REMARK VINA RESULT" in line:
                    fields = line.split()
                    if len(fields) >= 4:
                        score_value = fields[3]
                        with open(score_file_path, 'a') as score_file:
                            score_file.write(f"{folder_name} {score_value}\n")
                    break

    except Exception as e:
        with open(log_file_path, 'w') as log_file:
            log_file.write(f"Error processing {ligand_file}: {e}\n")
        print(f"Error processing {ligand_file}: {e}\n")
        
def sort_scores(file_path):
    # Read file content and compute average scores
    scores = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                name, score = parts
                try:
                    score = float(score)
                    scores.setdefault(name, []).append(score)
                except ValueError:
                    print(f"Invalid score value: {score}")

    # Compute average scores
    average_scores = [(name, np.mean(score_list)) for name, score_list in scores.items()]

    # Sort by average score in ascending order
    average_scores.sort(key=lambda x: x[1])
    
    # Write sorted results to file
    with open(file_path, 'w') as f:
        for name, avg_score in average_scores:
            f.write(f"{name} {avg_score}\n")

def main(args):
    np.random.seed(args.seed)
    
    if args.receptor is None:
        print("No receptor specified. Use '--receptor <receptor_file_name.pdbqt>'")
        exit(-1)

    # Directory containing ligand files
    ligand_dir = 'datasets/ligand_prep/'
    
    GB = GridBox("datasets/target_protein/native_lig.pdb")
    global center, bxsize
    center, bxsize = GB.labox()
    print('Center:', center)
    print('Box size:', bxsize)

    # Path for the score file
    score_file_path = 'docking_results/vina_score.txt'
    
    # Open the output file for scores
    with open(score_file_path, 'w') as score_file:
        # Loop through each ligand file in the directory
        ligand_files = [f for f in os.listdir(ligand_dir) if f.endswith('aligned.pdbqt')]
        
        # Use ProcessPoolExecutor to parallelize processing
        start_time = time.time()  # 记录开始时间
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_cores) as executor:
            futures = [executor.submit(process_ligand, ligand_file, ligand_dir, score_file_path, args.receptor) for ligand_file in ligand_files]
            
            # Wait for all futures to complete
            concurrent.futures.wait(futures)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算时间差
        print(f"Docking spent: {elapsed_time}seconds")
        
    # Sort the scores file by average score value in ascending order
    sort_scores(score_file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--receptor', default='datasets/target_protein/receptor.pdbqt', type=str, help='Processed receptor protein pdb file')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--num_cores', default=2, type=int, help='Number of CPU cores to use for parallel processing (default: 48)')
    
    args = parser.parse_args()
    main(args)
