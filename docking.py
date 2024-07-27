#!/usr/bin/env python
from vina import Vina
from argparse import ArgumentParser
import numpy as np
import random
import os
from grid_box import GridBox

parser = ArgumentParser()
parser.add_argument('--receptor', default='datasets/receptor.pdbqt', type=str, help='Processed recptor protein pdb file')
parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility (default: 42)')

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign


def main(args):
    np.random.seed(args.seed)
    # torch.random.manual_seed(args.seed)

    if args.receptor is None:
        print("No receptor specified. Use '--receptor <receptor_file_name.pdbqt>'")
        exit(-1)

    # Initialize Vina object
    v = Vina()

    # Set receptor
    v.set_receptor(rigid_pdbqt_filename="datasets/target_protein/receptor.pdbqt")

    # Directory containing ligand files
    ligand_dir = 'datasets/ligand_prep/'
    
    GB = GridBox("datasets/target_protein/native_lig.pdb")
    center, bxsize = GB.labox()
    print('center is ', center)
    print('box size is ', bxsize)
    v.compute_vina_maps(center=center, box_size=bxsize)
    # Loop through each ligand file in the directory
    for ligand_file in os.listdir(ligand_dir):
        if ligand_file.endswith('aligned.pdbqt'):
            ligand_path = os.path.join(ligand_dir, ligand_file)
            try:
                # Compute Vina maps
                # v.compute_vina_maps(center=center, box_size=bxsize)
                
                v.set_ligand_from_file(ligand_path)

                # Print score and optimization results
                print(f"Processing {ligand_file}:")
                print("Score:", v.score())
                print("Optimization:", v.optimize())

                # Perform docking
                v.dock(exhaustiveness=8)

                # Output file path
                output_path = f"docking_result/{ligand_file.replace('.pdbqt', '_docking_results.pdbqt')}"

                # Write poses to file
                v.write_poses(pdbqt_filename=output_path)
                print(f"Docking results saved to {output_path}\n")

            except Exception as e:
                print(f"Error processing {ligand_file}: {e}\n")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)