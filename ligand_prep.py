from meeko import MoleculePreparation
from meeko import PDBQTWriterLegacy
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

class LigandPrep:
    def __init__(self, smiles: str):
        """
        初始化LigandPrep类。
        
        :param smiles: 分子的SMILES字符串
        """
        self.smiles = smiles
        

    def smile2sdf(self, output_file: str):
        """
        准备分子并保存为sdf文件。
        
        :param output_file: 输出sdf文件的路径
        """
        # 添加3D坐标
        output_file_pdbqt = output_file.split(".")[0] + '.pdbqt'
        self.rdkit_mol = Chem.MolFromSmiles(self.smiles)
        self.rdkit_mol = Chem.AddHs(self.rdkit_mol)  # 添加氢原子
        if AllChem.EmbedMolecule(self.rdkit_mol) != 0:
            raise ValueError(f"无法为分子生成3D坐标: {smiles}")
        # AllChem.UFFOptimizeMolecule(self.rdkit_mol)  # 优化分子构象, UFF
        AllChem.MMFFOptimizeMolecule(self.rdkit_mol) # 优化分子构象
        Chem.MolToMolFile(self.rdkit_mol, output_file)
        
        # sdf to pdbqt
        # H atoms are merged
        sdf2pdbqt_prep = MoleculePreparation(merge_these_atom_types=()) #keep all hydrogen
        mol_setups = sdf2pdbqt_prep.prepare(self.rdkit_mol)
        for setup in mol_setups:
            pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
            if is_ok:
                # print(pdbqt_string, end="")
                with open(output_file_pdbqt, "w") as file:
                    file.write(pdbqt_string)
                    print(f"文件已保存到 {output_file_pdbqt}")
                