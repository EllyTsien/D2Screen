from meeko import MoleculePreparation
from meeko import PDBQTWriterLegacy
from rdkit import Chem
import numpy as np
from rdkit.Chem import AllChem, rdMolAlign, rdMolTransforms

class LigandPrep:
    def __init__(self, smiles: str):
        """
        初始化LigandPrep类。
        
        :param smiles: 分子的SMILES字符串
        """
        self.smiles = smiles
        
    def align_single_molecule(self, sdf_file, pdb_file, output_sdf_file):
        # Load the single molecule from the SDF file
        supplier = Chem.SDMolSupplier(sdf_file)
        mol = next(iter(supplier), None)
        if mol is None:
            raise ValueError("No valid molecule found in the SDF file")

        # Load the reference molecule from the PDB file
        ref_mol = Chem.MolFromPDBFile(pdb_file)
        if ref_mol is None:
            raise ValueError("Failed to load the reference molecule from the PDB file")

        # Ensure the molecule has a conformation
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol)
            
        # 计算参考分子的质心
        ref_conf = ref_mol.GetConformer()
        ref_centroid = rdMolTransforms.ComputeCentroid(ref_conf)
        mol_conf = mol.GetConformer()
        centroid = rdMolTransforms.ComputeCentroid(mol_conf)
        
        # 计算平移向量
        translation_vector = [ref_centroid.x - centroid.x, ref_centroid.y - centroid.y, ref_centroid.z - centroid.z]

        # 创建4x4的平移矩阵
        trans = np.eye(4)
        trans[0, 3] = translation_vector[0]
        trans[1, 3] = translation_vector[1]
        trans[2, 3] = translation_vector[2]

        # 应用变换
        AllChem.TransformConformer(mol_conf, trans)
        

        # Align the molecule to the reference molecule
        # rdMolAlign.AlignMol(mol, ref_mol)

        # Write the aligned molecule to the output SDF file
        writer = Chem.SDWriter(output_sdf_file)
        writer.write(mol)
        writer.close()

        print(f'Aligned molecule saved to {output_sdf_file}')
        

    def smile2sdf(self, output_file: str):
        """
        准备分子并保存为sdf文件, 再保存为pdbqt
        
        :param output_file: 输出sdf文件的路径
        """
        # 添加3D坐标
        output_file_pdbqt = output_file.split(".")[0] + '.pdbqt'
        output_file_aligned_sdf = output_file.split(".")[0] + '_aligned.sdf'
        output_file_aligned_pdbqt = output_file.split(".")[0] + '_aligned.pdbqt'
        self.rdkit_mol = Chem.MolFromSmiles(self.smiles)
        self.rdkit_mol = Chem.AddHs(self.rdkit_mol)  # 添加氢原子
        if AllChem.EmbedMolecule(self.rdkit_mol) != 0:
            raise ValueError(f"无法为分子生成3D坐标: {self.smiles}")
        # AllChem.UFFOptimizeMolecule(self.rdkit_mol)  # 优化分子构象, UFF
        AllChem.MMFFOptimizeMolecule(self.rdkit_mol) # 优化分子构象
        Chem.MolToMolFile(self.rdkit_mol, output_file)
        
        #align to native ligand
        self.align_single_molecule(output_file, "datasets/target_protein/native_lig.pdb", output_file_aligned_sdf)
        #读取align之后的分子
        supplier = Chem.SDMolSupplier(output_file_aligned_sdf)
        aligned_mol = next(supplier)
        aligned_mol = Chem.AddHs(aligned_mol)
        
        # sdf to pdbqt
        # H atoms are merged
        sdf2pdbqt_prep = MoleculePreparation(merge_these_atom_types=()) #keep all hydrogen
        mol_setups = sdf2pdbqt_prep.prepare(aligned_mol)
        for setup in mol_setups:
            pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
            if is_ok:
                # print(pdbqt_string, end="")
                with open(output_file_aligned_pdbqt, "w") as file:
                    file.write(pdbqt_string)
                    print(f"文件已保存到 {output_file_aligned_pdbqt}")
            else:
                print('transfer to pdbqt fail')
                