import torch
import pickle
from glob import glob
from tqdm import tqdm
from multiprocessing import Manager, Pool
from .AsmTokenizer import AsmLMTokenizer

class AsmLMDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, vocab_paths, n_workers=48):  
        self.n_workers = n_workers
        print('Initializing tokenizer...')
        self.tokenizer = AsmLMTokenizer(vocab_paths)

        print('Loading dataset...')
        self.dataset_path = dataset_path
        self.datas = Manager().list()
        self.load_data()

    def __getitem__(self, idx): # random visit
        return {key: torch.tensor(value) for key, value in self.datas[idx].items()}

    def __len__(self):
        return len(self.datas)

    def worker_load_data(self, pkl_file):
        pkl_datas = []
        with open(pkl_file, 'rb') as f:
            load = pickle.load(f)
            for func_info in tqdm(load):
                pkl_datas.append(self.tokenizer.encode_func(func_info)) # when parallel, save to tmpfs

        self.datas.extend(pkl_datas)
        del pkl_datas

    def load_data(self):
        pool = Pool(processes=self.n_workers)
        input_list = []
        for pkl_file in tqdm(glob('{}/*.pkl'.format(self.dataset_path))):
            input_list.append(pkl_file)
        pool.map(self.worker_load_data, input_list)
        pool.close()
        
        print(len(self.datas))

                        

class AsmLMPreGenDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, vocab_paths):  
        
        print('Initializing tokenizer...')
        self.tokenizer = AsmLMTokenizer(vocab_paths)

        print('Loading dataset...')
        self.dataset_path = dataset_path
        self.datas = []
        self.load_data()

    def __getitem__(self, idx): # random visit
        self.datas[idx]

    def __len__(self):
        return len(self.datas)

    def load_data(self):
        # TODO: load data
        pass


