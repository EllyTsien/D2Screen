import pandas as pd
from argparse import ArgumentParser

from ligand_prep import LigandPrep


# read top.csv
df = pd.read_csv("datasets/DL_pred/top.csv")
for index, row in df.iterrows():
    smiles = row['SMILES']
    ligand_id = row['ID']
    output_file = f"datasets/ligand_prep/ligand_{ligand_id}.sdf"
    
    try:
        ligand_prep = LigandPrep(smiles)
        ligand_prep.smile2sdf(output_file)
        print(f"分子已保存为 {output_file}")
    except ValueError as e:
        print(e)
        print(f"行 {index} 的 SMILES 无效: {smiles}")