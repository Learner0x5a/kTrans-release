"""
Configuration file.
"""
import os
import AsmLM

SEED = 42
EPOCH = 3
SAVED_MODEL_PATH = 'saved_models_pl'
# os.makedirs(SAVED_MODEL_PATH, exist_ok=True)

VOCAB_PATHS = {
    'token_vocab_path': f'{AsmLM.dataloader.__path__[0]}/token_vocab.txt',
    'itype_vocab_path': f'{AsmLM.dataloader.__path__[0]}/itype_vocab.txt',
    'opnd_type_vocab_path': f'{AsmLM.dataloader.__path__[0]}/opnd_type_vocab.txt',
    'reg_id_vocab_path': f'{AsmLM.dataloader.__path__[0]}/reg_id_vocab.txt',
    'reg_r_vocab_path': f'{AsmLM.dataloader.__path__[0]}/reg_r_vocab.txt',
    'reg_w_vocab_path': f'{AsmLM.dataloader.__path__[0]}/reg_w_vocab.txt',
    'eflags_vocab_path': f'{AsmLM.dataloader.__path__[0]}/eflags_vocab.txt',
}

TOKENIZER = AsmLM.AsmLMTokenizer(VOCAB_PATHS)

VOCAB_SIZES = {
    'token_vocab': len(TOKENIZER.token_vocab),
    'itype_vocab': len(TOKENIZER.itype_vocab),
    'opnd_type_vocab': len(TOKENIZER.opnd_type_vocab),
    'reg_id_vocab': len(TOKENIZER.reg_id_vocab),
    'reg_r_vocab': len(TOKENIZER.reg_r_vocab),
    'reg_w_vocab': len(TOKENIZER.reg_w_vocab),
    'eflags_vocab': len(TOKENIZER.eflags_vocab),
}

MIN_LENGTH = 4 # minimum length (# insns) of a function
NUM_GPUS = 4
ACCUMULATE_GRAD_BATCHES = 4
