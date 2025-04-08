import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import pickle as pkl
from multiprocessing import Process, Manager
import threading
import time
from pahelix.utils.compound_tools import mol_to_geognn_graph_data_MMFF3d

RDLogger.DisableLog('rdApp.*') # 屏蔽RDKit的warning

def worker_func(task_queue, smiles_to_graph_dict, invalid_smiles, counter, counter_lock):

    while True:
        try:
            smiles = task_queue.get_nowait()
        except Exception:
            break
        try:
            molecule = AllChem.MolFromSmiles(smiles)
            if molecule is None:
                # 如果解析失败，则记录为无效
                invalid_smiles.append(smiles)
            else:
                molecule_graph = mol_to_geognn_graph_data_MMFF3d(molecule)
                smiles_to_graph_dict[smiles] = molecule_graph
        except Exception:
            invalid_smiles.append(smiles)
        finally:
            with counter_lock:
                counter.value += 1


def progress_reporter_func(total_tasks, counter, counter_lock):
    while True:
        with counter_lock:
            done = counter.value
        percent = done / total_tasks * 100 if total_tasks else 0
        print(f"Progress: {done}/{total_tasks} ({percent:.2f}%)", end='\r')
        if done >= total_tasks:
            break
        time.sleep(1)
    print()


class Input_ligand_preprocess:
    def __init__(self, train_file):
        self.train_file = train_file

    def load_data(self):
        self.train_df = pd.read_csv(self.train_file)
        print(f'len of train_df is {len(self.train_df)}')

    def standardize_smiles(self):
        RDLogger.DisableLog('rdApp.*')
        for index, row in self.train_df.iterrows():
            try:
                mol = Chem.MolFromSmiles(row['SMILES'])
                new_smiles = Chem.MolToSmiles(mol)
                self.train_df.loc[index, 'SMILES'] = new_smiles
            except:
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

    def preprocess(self):
        self.load_data()
        self.standardize_smiles()
        self.remove_duplicates()
        self.save_processed_data()
        self.save_smiles_list()


class SMILES_Transfer:
    def __init__(self, train_df):
        self.train_df = train_df
        self.manager = Manager()
        self.task_queue = self.manager.Queue()
        self.smiles_to_graph_dict = self.manager.dict()
        self.invalid_smiles = self.manager.list()
        self.total_tasks = 0
        self.counter = self.manager.Value('i', 0)
        self.counter_lock = self.manager.Lock()

    def populate_tasks(self):
        smiles_list = self.train_df["SMILES"].tolist()
        self.total_tasks = len(smiles_list)
        for smiles in smiles_list:
            self.task_queue.put(smiles)

    def process_smiles(self, process_count=24):
        self.populate_tasks()
        progress_thread = threading.Thread(target=progress_reporter_func, args=(self.total_tasks, self.counter, self.counter_lock))
        progress_thread.start()
        processes = []
        for _ in range(process_count):
            p = Process(target=worker_func, args=(self.task_queue, self.smiles_to_graph_dict, self.invalid_smiles, self.counter, self.counter_lock))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        progress_thread.join()
        pkl.dump(dict(self.smiles_to_graph_dict), open('work/train_smiles_to_graph_dict.pkl', 'wb'))
        print('Train processing is Done!')
        if len(self.invalid_smiles) > 0:
            with open('work/invalid_smiles.txt', 'a') as f:
                for s in self.invalid_smiles:
                    f.write(s + '\n')

    def save_data_list(self):
        train_data_list = []
        smiles_to_graph = dict(self.smiles_to_graph_dict)
        for index, row in self.train_df.iterrows():
            smiles = row["SMILES"]
            label = row["label"]
            if smiles not in smiles_to_graph:
                continue
            data_item = {
                "smiles": smiles,
                "graph": smiles_to_graph[smiles],
                "label": label,
            }
            train_data_list.append(data_item)
        pkl.dump(train_data_list, open('work/train_data_list.pkl', 'wb'))
        print('Data lists have been saved!')

    def run(self):
        self.process_smiles()
        self.save_data_list()


if __name__ == "__main__":
    processor = Input_ligand_preprocess('datasets/input.csv')
    processor.preprocess()
    processed_input_csv = pd.read_csv('datasets/train_preprocessed.csv')
    SMILES_Transfer(processed_input_csv).run()
