import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import pickle as pkl
from threading import Thread, Lock
from pahelix.utils.compound_tools import mol_to_geognn_graph_data_MMFF3d


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


class SMILES_Transfer:
    def __init__(self, train_df):
    #def __init__(self, train_df, test_df):
        self.train_df = train_df
        # self.test_df = test_df
        self.mutex = Lock()
        self.p = 0
        self.smiles_to_graph_dict = {}

    def calculate_3D_structure(self, smiles_list):
        n = len(smiles_list)
        index = 0
        while True:
            self.mutex.acquire()
            if self.p >= n:
                self.mutex.release()
                break
            index = self.p
            smiles = smiles_list[index]
            print(index, ':', round(index / n * 100, 2), '%', smiles)
            self.p += 1
            self.mutex.release()
            
            try:
                molecule = AllChem.MolFromSmiles(smiles)
                molecule_graph = mol_to_geognn_graph_data_MMFF3d(molecule)
            except Exception as e:
                print("Invalid smiles!", smiles)
                self.mutex.acquire()
                with open('work/invalid_smiles.txt', 'a') as f:
                    f.write(str(smiles) + '\n')
                self.mutex.release()
                continue

            self.mutex.acquire()
            self.smiles_to_graph_dict[smiles] = molecule_graph
            self.mutex.release()

    def process_smiles(self, mode, thread_count=12):
        if mode == 'train':
            print(type(self.train_df))

            smiles_list = self.train_df["SMILES"].tolist()


        self.smiles_to_graph_dict = {}
        self.p = 0

        threads = []
        for _ in range(thread_count):
            threads.append(Thread(target=self.calculate_3D_structure, args=(smiles_list,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        pkl.dump(self.smiles_to_graph_dict, open(f'work/{mode}_smiles_to_graph_dict.pkl', 'wb'))
        print(f'{mode} is Done!')

    def save_data_list(self):
        train_smiles_to_graph_dict = pkl.load(open('work/train_smiles_to_graph_dict.pkl', 'rb'))
        # test_smiles_to_graph_dict = pkl.load(open('work/test_smiles_to_graph_dict.pkl', 'rb'))

        train_data_list = []
        # test_data_list = []

        for index, row in self.train_df.iterrows():
            smiles = row["SMILES"]
            label = row["label"]
            if smiles not in train_smiles_to_graph_dict:
                continue
            data_item = {
                "smiles": smiles,
                "graph": train_smiles_to_graph_dict[smiles],
                "label": label,
            }
            train_data_list.append(data_item)


        pkl.dump(train_data_list, open('work/train_data_list.pkl', 'wb'))
        # pkl.dump(test_data_list, open('work/test_data_list.pkl', 'wb'))
        print('Data lists have been saved!')

    def run(self):
        for mode in ['train']:
            self.process_smiles(mode)
        self.save_data_list()

# 使用示例
if __name__ == "__main__":
    processor = Input_ligand_preprocess('datasets/input.csv')
    processor.preprocess()
    processed_input_csv = pd.read_csv('datasets/train_preprocessed.csv')
    SMILES_Transfer = SMILES_Transfer(processed_input_csv)
    SMILES_Transfer.run()
