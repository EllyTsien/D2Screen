#!/usr/bin/env python
from vina import Vina
from argparse import ArgumentParser
import numpy as np
import os
from grid_box import GridBox
import sys
import contextlib

parser = ArgumentParser()
parser.add_argument('--receptor', default='datasets/target_protein/receptor.pdbqt', type=str, help='Processed receptor protein pdb file')
parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility (default: 42)')

def main(args):
    np.random.seed(args.seed)
    
    if args.receptor is None:
        print("No receptor specified. Use '--receptor <receptor_file_name.pdbqt>'")
        exit(-1)

    # Initialize Vina object
    v = Vina()

    # Set receptor
    v.set_receptor(rigid_pdbqt_filename=args.receptor)

    # Directory containing ligand files
    ligand_dir = 'datasets/ligand_prep/'
    
    GB = GridBox("datasets/target_protein/native_lig.pdb")
    center, bxsize = GB.labox()
    print('Center:', center)
    print('Box size:', bxsize)
    v.compute_vina_maps(center=center, box_size=bxsize)
    
    # Loop through each ligand file in the directory
    for ligand_file in os.listdir(ligand_dir):
        if ligand_file.endswith('aligned.pdbqt'):
            ligand_path = os.path.join(ligand_dir, ligand_file)
            
            # Remove 'aligned' suffix from folder name
            folder_name = ligand_file.replace('.pdbqt', '').replace('_aligned', '')
            folder_path = os.path.join('docking_results', folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            # Create log file path with ligand name
            log_file_path = os.path.join(folder_path, f'{folder_name}.log')

            try:
                v.set_ligand_from_file(ligand_path)

                # Get docking score and optimization results
                score = v.score()
                optimization = v.optimize()
                
                # Output file path
                output_path = os.path.join(folder_path, f'{folder_name}_docking_results.pdbqt')

                print(f'docking for {folder_name} start!')
                # Perform docking
                v.dock(exhaustiveness=8)
                print(f'docking for {folder_name} finished!')

                # Write poses to file
                v.write_poses(pdbqt_filename=output_path, n_poses=5, energy_range=3.0, overwrite=True)
                print(f"Docking results saved to {output_path}")
                print(f"Log information saved to {log_file_path}\n")

            except Exception as e:
                with open(log_file_path, 'w') as log_file:
                    log_file.write(f"Error processing {ligand_file}: {e}\n")
                print(f"Error processing {ligand_file}: {e}\n")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
