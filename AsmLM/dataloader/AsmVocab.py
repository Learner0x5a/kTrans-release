
class AsmLMVocab(object):
    def __init__(self, vocab:list=[]):
        self.vocab = ['[PAD]', '[SEP]', '[CLS]', '[UNK]', '[MASK]'] + vocab # NOTE: [PAD] must be the first word to generate padding masks for attention.
        self.pad_id = 0
        self.sep_id = 1
        self.cls_id = 2
        self.unk_id = 3
        self.mask_id = 4

        self.word2id = {token:idx for idx, token in enumerate(self.vocab)}
        self.id2word = {idx:token for idx, token in enumerate(self.vocab)}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, idx):
        return self.vocab[idx]

    def get_id(self, token):
        if token in self.vocab:
            return self.word2id[token]
        else:
            return self.unk_id

    def save(self, vocab_path):
        with open(vocab_path, 'w') as f:
            f.write('\n'.join(self.vocab))

    def load(self, vocab_path):
        with open(vocab_path, 'r') as f:
            self.vocab = f.read().split('\n')
            self.pad_id = self.vocab.index('[PAD]')
            self.sep_id = self.vocab.index('[SEP]')
            self.cls_id = self.vocab.index('[CLS]')
            self.unk_id = self.vocab.index('[UNK]')
            self.mask_id = self.vocab.index('[MASK]')
            
            self.word2id = {token:idx for idx, token in enumerate(self.vocab)}
            self.id2word = {idx:token for idx, token in enumerate(self.vocab)}

# if __name__ == '__main__':
#     import os
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     vocab = AsmLMVocab()
#     vocab.load(os.path.join(dir_path, 'vocab.txt'))
#     print(vocab)