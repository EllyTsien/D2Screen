import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import pickle as pkl
from multiprocessing import Process, Manager
import os
import time
import threading
from pahelix.utils.compound_tools import mol_to_geognn_graph_data_MMFF3d

# 屏蔽RDKit的warning
RDLogger.DisableLog('rdApp.*')

def worker_func(task_queue, smiles_to_graph_dict, invalid_smiles, counter, counter_lock):
    """
    从任务队列中获取 SMILES 字符串，转换为分子后计算图数据。
    转换失败的 SMILES 会记录到 invalid_smiles 列表中。
    每个任务完成后，counter 会递增，用于进度显示。
    """
    while True:
        try:
            smiles = task_queue.get_nowait()
        except Exception:
            break
        try:
            molecule = AllChem.MolFromSmiles(smiles)
            if molecule is None:
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
    """
    每秒打印一次处理进度，直到所有任务完成。
    """
    while True:
        with counter_lock:
            done = counter.value
        percent = done / total_tasks * 100 if total_tasks else 0
        print(f"Progress: {done}/{total_tasks} ({percent:.2f}%)", end='\r')
        if done >= total_tasks:
            break
        time.sleep(1)
    print()  # 换行

class Input_ligand_preprocess:
    def __init__(self, train_file, project_name, mode):
        self.train_file = train_file
        self.project_name = project_name
        self.mode = mode

    def load_data(self):
        if self.mode == 'train':
            self.train_df = pd.read_csv(self.train_file)
            print(f'len of train_df is {len(self.train_df)}')
        elif self.mode == 'DUDE':
            self.train_df = pd.read_csv(self.train_file)
            print(f'len of DUDE/{self.project_name} test_nolabel_df is {len(self.train_df)}')

    def standardize_smiles(self):
        RDLogger.DisableLog('rdApp.*')
        for index, row in self.train_df.iterrows():
            try:
                mol = Chem.MolFromSmiles(row['SMILES'])
                new_smiles = Chem.MolToSmiles(mol)
                self.train_df.loc[index, 'SMILES'] = new_smiles
            except Exception:
                self.train_df.drop(index, inplace=True)
        if self.mode == 'train':
            print(f'len of train_df after standardizing smiles is {len(self.train_df)}')
        elif self.mode == 'DUDE':
            print(f'len of DUDE/{self.project_name} test_nolabel_df after standardizing smiles is {len(self.train_df)}')

    def remove_duplicates(self):
        if self.mode == 'train':
            duplicate_rows = self.train_df[self.train_df.duplicated('SMILES', keep=False)]
            for smiles, group in duplicate_rows.groupby('SMILES'):
                if len(group.drop_duplicates(subset=['label'])) == 1:
                    self.train_df.drop(index=group.index[1:], inplace=True)
                else:
                    self.train_df.drop(index=group.index, inplace=True)    
                print(f'len of train_df after removing duplicates is {len(self.train_df)}')
        elif self.mode == 'DUDE':
            duplicate_rows = self.train_df[self.train_df.duplicated('SMILES', keep=False)]
            for smiles, group in duplicate_rows.groupby('SMILES'):
                self.train_df.drop(index=group.index[1:], inplace=True)
            print(f'len of DUDE/{self.project_name} test_nolabel_df after removing duplicates is {len(self.train_df)}')

    def save_processed_data(self):
        save_path = self.project_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if self.mode == 'train':
            self.train_df.to_csv(self.project_name + '/train_preprocessed.csv', index=False)
        elif self.mode == 'DUDE':
            self.train_df.to_csv(self.project_name + '/DUDE_test_nolabel_preprocessed.csv', index=False)

    def save_smiles_list(self):
        train_smiles_list = self.train_df["SMILES"].tolist()
        save_path = self.project_name + '/work/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if self.mode == 'train':
            pkl.dump(train_smiles_list, open(self.project_name + '/work/train_smiles_list.pkl', 'wb'))
        elif self.mode == 'DUDE':
            pkl.dump(train_smiles_list, open(self.project_name + '/work/DUDE_test_nolabel_smiles_list.pkl', 'wb'))

    def preprocess(self):
        self.load_data()
        self.standardize_smiles()
        self.remove_duplicates()
        self.save_processed_data()
        self.save_smiles_list()
        if self.mode == 'train':
            processed_input_path = self.project_name + '/train_preprocessed.csv'
        elif self.mode == 'DUDE':
            processed_input_path = self.project_name + '/DUDE_test_nolabel_preprocessed.csv'
        return processed_input_path

class SMILES_Transfer:
    def __init__(self, train_df, project_name, mode='train'):
        self.train_df = train_df
        self.project_name = project_name
        self.mode = mode
        # 使用Manager创建共享对象
        self.manager = Manager()
        self.smiles_to_graph_dict = self.manager.dict()
        self.invalid_smiles = self.manager.list()

    def process_smiles(self, process_count=12):
        smiles_list = self.train_df["SMILES"].tolist()
        total_tasks = len(smiles_list)
        print(f"Total tasks: {total_tasks}")

        # 使用Manager创建任务队列
        task_queue = self.manager.Queue()
        for smiles in smiles_list:
            task_queue.put(smiles)

        # 创建计数器和锁（用于进度显示）
        counter = self.manager.Value('i', 0)
        counter_lock = self.manager.Lock()

        # 开启进度显示线程
        progress_thread = threading.Thread(target=progress_reporter_func, args=(total_tasks, counter, counter_lock))
        progress_thread.start()

        # 启动多个工作进程
        processes = []
        for _ in range(process_count):
            p = Process(target=worker_func, args=(
                task_queue,
                self.smiles_to_graph_dict,
                self.invalid_smiles,
                counter,
                counter_lock
            ))
            p.daemon = True
            p.start()
            processes.append(p)
        # 等待所有子进程结束
        for p in processes:
            p.join()
        progress_thread.join()

        # 保存处理后的结果
        if self.mode == 'train':
            pkl.dump(self.smiles_to_graph_dict, open(self.project_name + '/work/train_smiles_to_graph_dict.pkl', 'wb'))
            print('Train processing is Done!')
            if len(self.invalid_smiles) > 0:
                with open(self.project_name + '/work/train_invalid_smiles.txt', 'a') as f:
                    for s in self.invalid_smiles:
                        f.write(s + '\n')
        elif self.mode == 'DUDE':
            pkl.dump(self.smiles_to_graph_dict, open(self.project_name + '/work/DUDE_nolabel_smiles_to_graph_dict.pkl', 'wb'))
            print('DUDE ' + self.project_name + ' test_nolabel preprocessing is Done!')
            if len(self.invalid_smiles) > 0:
                with open(self.project_name + '/work/DUDE_test_nolabel_invalid_smiles.txt', 'a') as f:
                    for s in self.invalid_smiles:
                        f.write(s + '\n')

    def save_data_list(self):
        if self.mode == 'train':
            train_data_list = []
            for index, row in self.train_df.iterrows():
                smiles = row["SMILES"]
                label = row["label"]
                if smiles not in self.smiles_to_graph_dict:
                    continue
                data_item = {
                    "smiles": smiles,
                    "graph": self.smiles_to_graph_dict[smiles],
                    "label": label,
                }
                train_data_list.append(data_item)
            pkl.dump(train_data_list, open(self.project_name + '/work/train_data_list.pkl', 'wb'))
        elif self.mode == 'DUDE':
            train_data_list = []
            for index, row in self.train_df.iterrows():
                smiles = row["SMILES"]
                if smiles not in self.smiles_to_graph_dict:
                    continue
                data_item = {
                    "smiles": smiles,
                    "graph": self.smiles_to_graph_dict[smiles],
                    "label": 0,
                }
                train_data_list.append(data_item)
            pkl.dump(train_data_list, open(self.project_name + '/work/DUDE_test_nolabel_data_list.pkl', 'wb'))
        print('Data lists have been saved!')

    def run(self):
        self.process_smiles()
        self.save_data_list()

if __name__ == "__main__":
    # 初始化预处理，生成标准化、去重后的csv文件
    processor = Input_ligand_preprocess('datasets/input.csv', project_name="example", mode='train')
    processor.preprocess()
    # 读取预处理后的csv文件，开始SMILES转换
    processed_input_csv = pd.read_csv('example/train_preprocessed.csv')
    SMILES_Transfer(processed_input_csv, project_name="example", mode='train').run()
