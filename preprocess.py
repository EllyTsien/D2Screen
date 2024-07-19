import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
import pickle as pkl
import paddle.nn as nn

class Input_ligand_preprocess:
    # def __init__(self, train_file, test_file):
    def __init__(self, train_file):
        self.train_file = train_file
        # self.test_file = test_file

    def load_data(self):
        self.train_df = pd.read_csv(self.train_file)
        # self.test_df = pd.read_csv(self.test_file)
        print(f'len of train_df is {len(self.train_df)}')

    def standardize_smiles(self):
        RDLogger.DisableLog('rdApp.*') # 屏蔽RDKit的warning

        for index, row in self.train_df.iterrows():
            try:
                mol = Chem.MolFromSmiles(row['SMILES'])
                new_smiles = Chem.MolToSmiles(mol)
                self.train_df.loc[index, 'SMILES'] = new_smiles
            except:
                # 若转化失败，则认为原始smile不合法，删除该数据
                self.train_df.drop(index, inplace=True)

        print(f'len of train_df after standardizing smiles is {len(self.train_df)}')

    def remove_duplicates(self):
        duplicate_rows = self.train_df[self.train_df.duplicated('SMILES', keep=False)]

        for smiles, group in duplicate_rows.groupby('SMILES'):
            if len(group.drop_duplicates(subset=['label'])) == 1:
                self.train_df.drop(index=group.index[1:], inplace=True)
            else:
                self.train_df.drop(index=group.index, inplace=True)

        print(f'len of train_df after removing duplicates is {len(self.train_df)}')

    def save_processed_data(self):
        self.train_df.to_csv('datasets/train_preprocessed.csv', index=0)

    def save_smiles_list(self):
        train_smiles_list = self.train_df["SMILES"].tolist()
        pkl.dump(train_smiles_list, open('work/train_smiles_list.pkl', 'wb'))

        #test_smiles_list = self.test_df["SMILES"].tolist()
        # pkl.dump(test_smiles_list, open('work/test_smiles_list.pkl', 'wb'))

    def preprocess(self):
        self.load_data()
        self.standardize_smiles()
        self.remove_duplicates()
        self.save_processed_data()
        self.save_smiles_list()

# 使用示例
if __name__ == "__main__":
    processor = Input_ligand_preprocess('datasets/input.csv')
    processor.preprocess()
