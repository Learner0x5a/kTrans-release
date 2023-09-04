
import re
from typing import Tuple
# import torch
import numpy as np
from .AsmVocab import AsmLMVocab

class AsmLMTokenizer:
    def __init__(self, vocab_paths, mlm_probability=0.15, max_len=512):

        self.token_vocab = AsmLMVocab()
        self.itype_vocab = AsmLMVocab()
        self.opnd_type_vocab = AsmLMVocab()
        self.reg_id_vocab = AsmLMVocab()
        self.reg_r_vocab = AsmLMVocab()
        self.reg_w_vocab = AsmLMVocab()
        self.eflags_vocab = AsmLMVocab()
        
        self.token_vocab.load(vocab_paths['token_vocab_path'])
        self.itype_vocab.load(vocab_paths['itype_vocab_path'])
        self.opnd_type_vocab.load(vocab_paths['opnd_type_vocab_path'])
        self.reg_id_vocab.load(vocab_paths['reg_id_vocab_path'])
        self.reg_r_vocab.load(vocab_paths['reg_r_vocab_path'])
        self.reg_w_vocab.load(vocab_paths['reg_w_vocab_path'])
        self.eflags_vocab.load(vocab_paths['eflags_vocab_path'])


        self.mlm_probability = mlm_probability
        self.max_len = max_len
    
    def normalize_insn(self, asm):
        opcode, op_str = asm.split('\t')
        
        op_str = op_str.replace(' + ', '+')
        op_str = op_str.replace(' - ', '-')
        op_str = op_str.replace(' * ', '*')
        op_str = op_str.replace(' : ', ':')
        # op_str = op_str.replace(',', ' ,')
        op_str = re.sub('0x[0-9a-f]+', 'const', op_str)
        # print(f'{opcode} {op_str}')
        if op_str:
            opnd_strs = op_str.split(', ')
        else:
            opnd_strs = []

        return opcode, opnd_strs

    
    def encode_insn(self, insn_asm, itype, op_info, eflags):

        opcode, opnd_strs = self.normalize_insn(insn_asm)
        opcode_tokens = opcode.split()
        insn_tokens = opcode_tokens + ' , '.join(opnd_strs).split() # -> List[str]

        token_ids = [self.token_vocab.get_id(x) for x in insn_tokens]

        
        insn_type_ids = [self.itype_vocab.get_id(str(itype))] * len(insn_tokens)
        eflags_ids = [self.eflags_vocab.get_id(str(eflags))] * len(insn_tokens)

        if len(op_info) < len(opnd_strs): 
            print('op_info less than opnd_strs!') 
            print(insn_asm,itype,op_info)
            print(opcode, opnd_strs)
            # exit(1)
            return None 
        
        opnd_type_ids = [0] * len(opcode_tokens)
        reg_id_ids = [0] * len(opcode_tokens)
        opnd_r_ids = [0] * len(opcode_tokens)
        opnd_w_ids = [0] * len(opcode_tokens)

        '''special instruction such as nop mul, cdqe that have implicit operands
        nop -> xchg eax, eax
        mul rdx -> ax = rdx*ax
        cdqe -> rax = eax
        '''
        for i in range(len(opnd_strs)): 
            
            opnd_type_ids += [self.opnd_type_vocab.get_id(str(op_info[i][0]))] * len(opnd_strs[i].split()) + [self.opnd_type_vocab.pad_id] # pad for the comma
            reg_id_ids += [self.reg_id_vocab.get_id(str(op_info[i][1]))] * len(opnd_strs[i].split()) + [self.reg_id_vocab.pad_id]
            opnd_r_ids += [self.reg_r_vocab.get_id(str(op_info[i][2]))] * len(opnd_strs[i].split()) + [self.reg_r_vocab.pad_id]
            opnd_w_ids += [self.reg_w_vocab.get_id(str(op_info[i][3]))] * len(opnd_strs[i].split()) + [self.reg_w_vocab.pad_id]
        
        if opnd_strs: # if operand token exists, pop the extra 0 in the end
            opnd_type_ids.pop() 
            reg_id_ids.pop()
            opnd_r_ids.pop()
            opnd_w_ids.pop()


        if len(opnd_type_ids) != len(insn_tokens):
            print('opnd embedding should have same dimensions as token embedding')
            print(insn_asm,itype,op_info)
            print(insn_tokens)
            print(insn_type_ids)
            print(opnd_type_ids)
            print(reg_id_ids)
            print(opnd_r_ids)
            print(opnd_w_ids)
            # exit(2)
            return None

        
        result = {
            'token_ids': token_ids, 
            'insn_type_ids': insn_type_ids,
            'opnd_type_ids': opnd_type_ids,
            'reg_id_ids': reg_id_ids,
            'opnd_r_ids': opnd_r_ids,
            'opnd_w_ids': opnd_w_ids,
            'eflags_ids': eflags_ids
        }

        return result
    
    def encode_func_iter(self, func_info):
        '''
        0. get raw data
        1. tokenize insn asm
        2. generate masks/labels for asm tokens
        3. add special tokens
        4. padding
        5. optional: additional labels for other pretrain tasks. 
        '''

        func_token_ids = []
        func_insn_type_ids = []
        func_opnd_type_ids = []
        func_reg_id_ids = []
        func_opnd_r_ids = []
        func_opnd_w_ids = []
        func_eflags_ids = []

        entity_masked_func_token_ids = []
        entity_masked_func_token_labels = []

        # merge insn encodings
        for disasm,itype,op_info,eflags in func_info:
            if not disasm:
                continue
            encodings = self.encode_insn(disasm,itype,op_info,eflags)
            if encodings is not None:

                func_token_ids.extend(encodings['token_ids'])
                func_insn_type_ids.extend(encodings['insn_type_ids'])
                func_opnd_type_ids.extend(encodings['opnd_type_ids'])
                func_reg_id_ids.extend(encodings['reg_id_ids'])
                func_opnd_r_ids.extend(encodings['opnd_r_ids'])
                func_opnd_w_ids.extend(encodings['opnd_w_ids'])
                func_eflags_ids.extend(encodings['eflags_ids'])

                # add instruction-level mask
                # 15% to mask
                insn_mask = np.random.binomial(1, self.mlm_probability)
                if insn_mask:
                    entity_masked_func_token_ids.extend(len(encodings['token_ids']) * [self.token_vocab.mask_id])
                    # for masked tokens, labels are set to token ids
                    entity_masked_func_token_labels.extend(encodings['token_ids'])
                else:
                    entity_masked_func_token_ids.extend(encodings['token_ids'])
                    # for unmasked tokens, labels are set to [PAD], which will be ignored when calculating the loss function
                    entity_masked_func_token_labels.extend(len(encodings['token_ids']) * [self.token_vocab.pad_id])

        
        # vanilla labels are token ids themselves
        func_token_labels = np.copy(np.array(func_token_ids)).tolist()

        # generate masks/labels for asm tokens
        masked_func_token_ids, masked_func_token_labels = self.mask_tokens(func_token_ids)
        masked_func_token_ids = masked_func_token_labels.tolist()
        masked_func_token_labels = masked_func_token_labels.tolist()
        
        # add special tokens 
        func_token_ids = ([self.token_vocab.cls_id] + func_token_ids + [self.token_vocab.sep_id])[:self.max_len]
        func_token_labels = ([self.token_vocab.cls_id] + func_token_labels + [self.token_vocab.sep_id])[:self.max_len]
        masked_func_token_ids = ([self.token_vocab.cls_id] + masked_func_token_ids + [self.token_vocab.sep_id])[:self.max_len]
        masked_func_token_labels = ([self.token_vocab.pad_id] + masked_func_token_labels + [self.token_vocab.pad_id])[:self.max_len]
        func_insn_type_ids = ([self.itype_vocab.cls_id] + func_insn_type_ids + [self.itype_vocab.sep_id])[:self.max_len]
        func_opnd_type_ids = ([self.opnd_type_vocab.cls_id] + func_opnd_type_ids + [self.opnd_type_vocab.sep_id])[:self.max_len]
        func_reg_id_ids = ([self.reg_id_vocab.cls_id] + func_reg_id_ids + [self.reg_id_vocab.sep_id])[:self.max_len]
        func_opnd_r_ids = ([self.reg_r_vocab.cls_id] + func_opnd_r_ids + [self.reg_r_vocab.sep_id])[:self.max_len]
        func_opnd_w_ids = ([self.reg_w_vocab.cls_id] + func_opnd_w_ids + [self.reg_w_vocab.sep_id])[:self.max_len]
        func_eflags_ids = ([self.eflags_vocab.cls_id] + func_eflags_ids + [self.eflags_vocab.sep_id])[:self.max_len]
        entity_masked_func_token_ids = ([self.token_vocab.cls_id] + entity_masked_func_token_ids + [self.token_vocab.sep_id])[:self.max_len]
        entity_masked_func_token_labels = ([self.token_vocab.pad_id] + entity_masked_func_token_labels + [self.token_vocab.pad_id])[:self.max_len]
        

        outputs =  {
            'func_token_ids': func_token_ids,
            'func_token_labels': func_token_labels,
            'masked_func_token_ids': masked_func_token_ids,
            'masked_func_token_labels': masked_func_token_labels,
            'func_insn_type_ids': func_insn_type_ids,
            'func_opnd_type_ids': func_opnd_type_ids,
            'func_reg_id_ids': func_reg_id_ids,
            'func_opnd_r_ids': func_opnd_r_ids,
            'func_opnd_w_ids': func_opnd_w_ids,
            'func_eflags_ids': func_eflags_ids,
            'entity_masked_func_token_ids':entity_masked_func_token_ids,
            'entity_masked_func_token_labels':entity_masked_func_token_labels

        }

        return outputs

    def encode_func_with_insn_boundary(self, func_info):
        '''
        0. get raw data
        1. tokenize insn asm
        2. generate masks/labels for asm tokens
        3. add special tokens
        4. padding
        5. optional: additional labels for other pretrain tasks. 
        '''

        func_token_ids = []
        func_insn_type_ids = []
        func_opnd_type_ids = []
        func_reg_id_ids = []
        func_opnd_r_ids = []
        func_opnd_w_ids = []
        func_eflags_ids = []

        entity_masked_func_token_ids = []
        entity_masked_func_token_labels = []

        func_insn_boundary_ids = []

        # merge insn encodings
        insn_idx = 0
        for insn_addr,disasm,itype,op_info,eflags in func_info:
            if not disasm:
                continue
            encodings = self.encode_insn(disasm,itype,op_info,eflags)
            if encodings is not None:
                func_token_ids.extend(encodings['token_ids'])
                func_insn_type_ids.extend(encodings['insn_type_ids'])
                func_opnd_type_ids.extend(encodings['opnd_type_ids'])
                func_reg_id_ids.extend(encodings['reg_id_ids'])
                func_opnd_r_ids.extend(encodings['opnd_r_ids'])
                func_opnd_w_ids.extend(encodings['opnd_w_ids'])
                func_eflags_ids.extend(encodings['eflags_ids'])
        
                # add instruction-level mask
                # 15% to mask
                insn_mask = np.random.binomial(1, self.mlm_probability)
                if insn_mask:
                    entity_masked_func_token_ids.extend(len(encodings['token_ids']) * [self.token_vocab.mask_id])
                    # for masked tokens, labels are set to token ids
                    entity_masked_func_token_labels.extend(encodings['token_ids'])
                else:
                    entity_masked_func_token_ids.extend(encodings['token_ids'])
                    # for unmasked tokens, labels are set to [PAD], which will be ignored when calculating the loss function
                    entity_masked_func_token_labels.extend(len(encodings['token_ids']) * [self.token_vocab.pad_id])

                # add instruction boundary
                func_insn_boundary_ids.extend(len(encodings['token_ids']) * [insn_idx])
                insn_idx += 1


        # vanilla labels are token ids themselves
        func_token_labels = np.copy(np.array(func_token_ids)).tolist()

        # generate masks/labels for asm tokens
        masked_func_token_ids, masked_func_token_labels = self.mask_tokens(func_token_ids)
        masked_func_token_ids = masked_func_token_labels.tolist()
        masked_func_token_labels = masked_func_token_labels.tolist()
        
        # add special tokens 
        func_token_ids = ([self.token_vocab.cls_id] + func_token_ids + [self.token_vocab.sep_id])[:self.max_len]
        func_token_labels = ([self.token_vocab.cls_id] + func_token_labels + [self.token_vocab.sep_id])[:self.max_len]
        masked_func_token_ids = ([self.token_vocab.cls_id] + masked_func_token_ids + [self.token_vocab.sep_id])[:self.max_len]
        # special token labels are set to [PAD]
        masked_func_token_labels = ([self.token_vocab.pad_id] + masked_func_token_labels + [self.token_vocab.pad_id])[:self.max_len]
        func_insn_type_ids = ([self.itype_vocab.cls_id] + func_insn_type_ids + [self.itype_vocab.sep_id])[:self.max_len]
        func_opnd_type_ids = ([self.opnd_type_vocab.cls_id] + func_opnd_type_ids + [self.opnd_type_vocab.sep_id])[:self.max_len]
        func_reg_id_ids = ([self.reg_id_vocab.cls_id] + func_reg_id_ids + [self.reg_id_vocab.sep_id])[:self.max_len]
        func_opnd_r_ids = ([self.reg_r_vocab.cls_id] + func_opnd_r_ids + [self.reg_r_vocab.sep_id])[:self.max_len]
        func_opnd_w_ids = ([self.reg_w_vocab.cls_id] + func_opnd_w_ids + [self.reg_w_vocab.sep_id])[:self.max_len]
        func_eflags_ids = ([self.eflags_vocab.cls_id] + func_eflags_ids + [self.eflags_vocab.sep_id])[:self.max_len]
        entity_masked_func_token_ids = ([self.token_vocab.cls_id] + entity_masked_func_token_ids + [self.token_vocab.sep_id])[:self.max_len]
        # special token labels are set to [PAD]
        entity_masked_func_token_labels = ([self.token_vocab.pad_id] + entity_masked_func_token_labels + [self.token_vocab.pad_id])[:self.max_len]
    
        func_insn_boundary_ids = ([-1] + func_insn_boundary_ids + [-1])[:self.max_len]

        # padding
        padding_func_token_ids = [self.token_vocab.pad_id for _ in range(self.max_len - len(func_token_ids))]
        padding_func_token_labels = [self.token_vocab.pad_id for _ in range(self.max_len - len(func_token_labels))]
        padding_masked_func_token_ids = [self.token_vocab.pad_id for _ in range(self.max_len - len(masked_func_token_ids))]
        padding_masked_func_token_labels = [self.token_vocab.pad_id for _ in range(self.max_len - len(masked_func_token_labels))]
        padding_func_insn_type_ids = [self.token_vocab.pad_id for _ in range(self.max_len - len(func_insn_type_ids))]
        padding_func_opnd_type_ids = [self.token_vocab.pad_id for _ in range(self.max_len - len(func_opnd_type_ids))]
        padding_func_reg_id_ids = [self.token_vocab.pad_id for _ in range(self.max_len - len(func_reg_id_ids))]
        padding_func_opnd_r_ids = [self.token_vocab.pad_id for _ in range(self.max_len - len(func_opnd_r_ids))]
        padding_func_opnd_w_ids = [self.token_vocab.pad_id for _ in range(self.max_len - len(func_opnd_w_ids))]
        padding_func_eflags_ids = [self.token_vocab.pad_id for _ in range(self.max_len - len(func_eflags_ids))]
        padding_entity_masked_func_token_ids = [self.token_vocab.pad_id for _ in range(self.max_len - len(entity_masked_func_token_ids))]
        padding_entity_masked_func_token_labels = [self.token_vocab.pad_id for _ in range(self.max_len - len(entity_masked_func_token_labels))]
        padding_func_insn_boundary_ids = [-100 for _ in range(self.max_len - len(func_insn_boundary_ids))]

        func_token_ids.extend(padding_func_token_ids)
        func_token_labels.extend(padding_func_token_labels)
        masked_func_token_ids.extend(padding_masked_func_token_ids)
        masked_func_token_labels.extend(padding_masked_func_token_labels)
        func_insn_type_ids.extend(padding_func_insn_type_ids)
        func_opnd_type_ids.extend(padding_func_opnd_type_ids)
        func_reg_id_ids.extend(padding_func_reg_id_ids)
        func_opnd_r_ids.extend(padding_func_opnd_r_ids)
        func_opnd_w_ids.extend(padding_func_opnd_w_ids)
        func_eflags_ids.extend(padding_func_eflags_ids)
        entity_masked_func_token_ids.extend(padding_entity_masked_func_token_ids)
        entity_masked_func_token_labels.extend(padding_entity_masked_func_token_labels)
        func_insn_boundary_ids.extend(padding_func_insn_boundary_ids)

        outputs =  {
            'func_token_ids': func_token_ids,
            'func_token_labels': func_token_labels,
            'masked_func_token_ids': masked_func_token_ids,
            'masked_func_token_labels': masked_func_token_labels,
            'func_insn_type_ids': func_insn_type_ids,
            'func_opnd_type_ids': func_opnd_type_ids,
            'func_reg_id_ids': func_reg_id_ids,
            'func_opnd_r_ids': func_opnd_r_ids,
            'func_opnd_w_ids': func_opnd_w_ids,
            'func_eflags_ids': func_eflags_ids,
            'entity_masked_func_token_ids':entity_masked_func_token_ids,
            'entity_masked_func_token_labels':entity_masked_func_token_labels,
            'func_insn_boundary_ids':func_insn_boundary_ids,
        }

        return outputs




    def encode_func(self, func_info):
        '''
        0. get raw data
        1. tokenize insn asm
        2. generate masks/labels for asm tokens
        3. add special tokens
        4. padding
        5. optional: additional labels for other pretrain tasks. 
        '''

        func_token_ids = []
        func_insn_type_ids = []
        func_opnd_type_ids = []
        func_reg_id_ids = []
        func_opnd_r_ids = []
        func_opnd_w_ids = []
        func_eflags_ids = []

        entity_masked_func_token_ids = []
        entity_masked_func_token_labels = []

        # merge insn encodings
        for insn_addr,disasm,itype,op_info,eflags in func_info:
            if not disasm:
                continue
            encodings = self.encode_insn(disasm,itype,op_info,eflags)
            if encodings is not None:
                func_token_ids.extend(encodings['token_ids'])
                func_insn_type_ids.extend(encodings['insn_type_ids'])
                func_opnd_type_ids.extend(encodings['opnd_type_ids'])
                func_reg_id_ids.extend(encodings['reg_id_ids'])
                func_opnd_r_ids.extend(encodings['opnd_r_ids'])
                func_opnd_w_ids.extend(encodings['opnd_w_ids'])
                func_eflags_ids.extend(encodings['eflags_ids'])
        
                # add instruction-level mask
                # 15% to mask
                insn_mask = np.random.binomial(1, self.mlm_probability)
                if insn_mask:
                    entity_masked_func_token_ids.extend(len(encodings['token_ids']) * [self.token_vocab.mask_id])
                    # for masked tokens, labels are set to token ids
                    entity_masked_func_token_labels.extend(encodings['token_ids'])
                else:
                    entity_masked_func_token_ids.extend(encodings['token_ids'])
                    # for unmasked tokens, labels are set to [PAD], which will be ignored when calculating the loss function
                    entity_masked_func_token_labels.extend(len(encodings['token_ids']) * [self.token_vocab.pad_id])

        # vanilla labels are token ids themselves
        func_token_labels = np.copy(np.array(func_token_ids)).tolist()

        # generate masks/labels for asm tokens
        masked_func_token_ids, masked_func_token_labels = self.mask_tokens(func_token_ids)
        masked_func_token_ids = masked_func_token_labels.tolist()
        masked_func_token_labels = masked_func_token_labels.tolist()
        
        # add special tokens 
        func_token_ids = ([self.token_vocab.cls_id] + func_token_ids + [self.token_vocab.sep_id])[:self.max_len]
        func_token_labels = ([self.token_vocab.cls_id] + func_token_labels + [self.token_vocab.sep_id])[:self.max_len]
        masked_func_token_ids = ([self.token_vocab.cls_id] + masked_func_token_ids + [self.token_vocab.sep_id])[:self.max_len]
        # special token labels are set to [PAD]
        masked_func_token_labels = ([self.token_vocab.pad_id] + masked_func_token_labels + [self.token_vocab.pad_id])[:self.max_len]
        func_insn_type_ids = ([self.itype_vocab.cls_id] + func_insn_type_ids + [self.itype_vocab.sep_id])[:self.max_len]
        func_opnd_type_ids = ([self.opnd_type_vocab.cls_id] + func_opnd_type_ids + [self.opnd_type_vocab.sep_id])[:self.max_len]
        func_reg_id_ids = ([self.reg_id_vocab.cls_id] + func_reg_id_ids + [self.reg_id_vocab.sep_id])[:self.max_len]
        func_opnd_r_ids = ([self.reg_r_vocab.cls_id] + func_opnd_r_ids + [self.reg_r_vocab.sep_id])[:self.max_len]
        func_opnd_w_ids = ([self.reg_w_vocab.cls_id] + func_opnd_w_ids + [self.reg_w_vocab.sep_id])[:self.max_len]
        func_eflags_ids = ([self.eflags_vocab.cls_id] + func_eflags_ids + [self.eflags_vocab.sep_id])[:self.max_len]
        entity_masked_func_token_ids = ([self.token_vocab.cls_id] + entity_masked_func_token_ids + [self.token_vocab.sep_id])[:self.max_len]
        # special token labels are set to [PAD]
        entity_masked_func_token_labels = ([self.token_vocab.pad_id] + entity_masked_func_token_labels + [self.token_vocab.pad_id])[:self.max_len]

        # padding
        padding_func_token_ids = [self.token_vocab.pad_id for _ in range(self.max_len - len(func_token_ids))]
        padding_func_token_labels = [self.token_vocab.pad_id for _ in range(self.max_len - len(func_token_labels))]
        padding_masked_func_token_ids = [self.token_vocab.pad_id for _ in range(self.max_len - len(masked_func_token_ids))]
        padding_masked_func_token_labels = [self.token_vocab.pad_id for _ in range(self.max_len - len(masked_func_token_labels))]
        padding_func_insn_type_ids = [self.token_vocab.pad_id for _ in range(self.max_len - len(func_insn_type_ids))]
        padding_func_opnd_type_ids = [self.token_vocab.pad_id for _ in range(self.max_len - len(func_opnd_type_ids))]
        padding_func_reg_id_ids = [self.token_vocab.pad_id for _ in range(self.max_len - len(func_reg_id_ids))]
        padding_func_opnd_r_ids = [self.token_vocab.pad_id for _ in range(self.max_len - len(func_opnd_r_ids))]
        padding_func_opnd_w_ids = [self.token_vocab.pad_id for _ in range(self.max_len - len(func_opnd_w_ids))]
        padding_func_eflags_ids = [self.token_vocab.pad_id for _ in range(self.max_len - len(func_eflags_ids))]
        padding_entity_masked_func_token_ids = [self.token_vocab.pad_id for _ in range(self.max_len - len(entity_masked_func_token_ids))]
        padding_entity_masked_func_token_labels = [self.token_vocab.pad_id for _ in range(self.max_len - len(entity_masked_func_token_labels))]

        func_token_ids.extend(padding_func_token_ids)
        func_token_labels.extend(padding_func_token_labels)
        masked_func_token_ids.extend(padding_masked_func_token_ids)
        masked_func_token_labels.extend(padding_masked_func_token_labels)
        func_insn_type_ids.extend(padding_func_insn_type_ids)
        func_opnd_type_ids.extend(padding_func_opnd_type_ids)
        func_reg_id_ids.extend(padding_func_reg_id_ids)
        func_opnd_r_ids.extend(padding_func_opnd_r_ids)
        func_opnd_w_ids.extend(padding_func_opnd_w_ids)
        func_eflags_ids.extend(padding_func_eflags_ids)
        entity_masked_func_token_ids.extend(padding_entity_masked_func_token_ids)
        entity_masked_func_token_labels.extend(padding_entity_masked_func_token_labels)

        outputs =  {
            'func_token_ids': func_token_ids,
            'func_token_labels': func_token_labels,
            'masked_func_token_ids': masked_func_token_ids,
            'masked_func_token_labels': masked_func_token_labels,
            'func_insn_type_ids': func_insn_type_ids,
            'func_opnd_type_ids': func_opnd_type_ids,
            'func_reg_id_ids': func_reg_id_ids,
            'func_opnd_r_ids': func_opnd_r_ids,
            'func_opnd_w_ids': func_opnd_w_ids,
            'func_eflags_ids': func_eflags_ids,
            'entity_masked_func_token_ids':entity_masked_func_token_ids,
            'entity_masked_func_token_labels':entity_masked_func_token_labels
        }

        return outputs
        # return {key: torch.tensor(value) for key, value in outputs.items()}


    def mask_tokens(self, inputs):
        '''Adopted and adapted from HuggingFace Transformers
        The MLM task: for non-masked tokens, lables are -100; for masked tokens, labels are the input_ids to predict
        '''
        inputs = np.array(inputs)
        labels = np.copy(inputs)
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = np.full(labels.shape, self.mlm_probability)

        # Numpy doesn't have bernoulli, so we use a binomial with 1 trial
        masked_indices = np.random.binomial(1, probability_matrix, size=probability_matrix.shape).astype(bool)
        # For unmasked tokens, we set the labels to [PAD], which will be ignored when calculating the loss function
        # with the `ignore_index` parameter: https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
        labels[~masked_indices] = self.token_vocab.pad_id 

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = np.random.binomial(1, 0.8, size=labels.shape).astype(bool) & masked_indices
        inputs[indices_replaced] = self.token_vocab.mask_id

        # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        indices_random = (
            np.random.binomial(1, 0.5, size=labels.shape).astype(bool) & masked_indices & ~indices_replaced
        )
        random_words = np.random.randint(
            low=0, high=len(self.token_vocab.vocab), size=np.count_nonzero(indices_random), dtype=np.int64
        )
        inputs[indices_random] = random_words

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
