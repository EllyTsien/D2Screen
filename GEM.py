import pickle as pkl
from threading import Thread, Lock
from pahelix.utils.compound_tools import mol_to_geognn_graph_data_MMFF3d
from rdkit.Chem import AllChem

class GEM_smile_transfer:
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
    import pandas as pd

    
    
    # test_df = pd.read_csv('data/data221048/test_nolabel.csv')

    gem_smile_transfer = GEM_smile_transfer(train_df)
    gem_smile_transfer.run()
