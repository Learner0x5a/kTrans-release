import os
import pickle
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from asmlm import *

class FuncDataset(torch.utils.data.Dataset): 
    def __init__(self, path): 
        self.path = path
        assert os.path.exists(self.path), "Dataset Path Not Exists"
        self.files = glob(os.path.join(self.path, '*.pkl'))

        self.datas = []
        for file in self.files:
            with open(file, 'rb') as f:
                FUNC_INFO = pickle.load(f)
                for func_info in tqdm(FUNC_INFO, desc='Loading file %s' % file, ncols=80, ascii=' #'):
                    encodings = TOKENIZER.encode_func(func_info)
                    encodings = {key: torch.tensor(value) for key, value in encodings.items()}
                    self.datas.append(encodings)

    def __getitem__(self, idx):
        return self.datas[idx]
        
    def __len__(self):
        return len(self.datas)

if __name__ == "__main__":

    parser = ArgumentParser(description="Generate embeddings with pretrained/finetuned kTrans model")
    parser.add_argument("-i", "--input_dir", help="Input dir of pregen pairs", default='./ida_outputs', nargs='?')                              
    parser.add_argument("-m", "--model_path", help="path to kTrans model", default='./ktrans-110M-epoch-2.ckpt', nargs='?')
    parser.add_argument("-n", "--num_process", help="num of workers for dataloader", default=64, nargs='?')
    parser.add_argument("-bs", "--batch_size", help="batch size", default=128, nargs='?')  
    parser.add_argument("-o", "--output_dir", help="output directory", default='ktrans-saved_embs', nargs='?')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = FuncDataset(args.input_dir)
    dl = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_process))

    
    print('Loading kTrans model...')
    pl_model = AsmLMModule.load_from_checkpoint("{}".format(args.model_path), total_steps=0)
    bert_model = pl_model.bert

    dev = torch.device('cuda')
    bert_model.to(dev)
    # bert_model = torch.nn.DataParallel(bert_model) # enable this if you use multi-gpu
    bert_model.eval()


    with torch.no_grad():
        EMBEDDINGS = []
        for idx, batch in tqdm(enumerate(dl), desc='Generating Embeddings', ncols=80, ascii=' #'):

            func_emb = bert_model(
                batch['func_token_ids'].to(dev),
                batch['func_insn_type_ids'].to(dev),
                batch['func_opnd_type_ids'].to(dev),
                batch['func_reg_id_ids'].to(dev),
                batch['func_opnd_r_ids'].to(dev),
                batch['func_opnd_w_ids'].to(dev),
                batch['func_eflags_ids'].to(dev),
            )[:,0,:].cpu().detach().numpy()

            EMBEDDINGS.append(func_emb)

        EMBEDDINGS = np.concatenate(EMBEDDINGS, axis=0)
        with open(f'{args.output_dir}/emb.pkl', 'wb') as f:
            pickle.dump(EMBEDDINGS, f)


        