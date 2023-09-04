import torch.nn as nn
import torch

from .bert import BERT


class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, token_vocab_size):
        """
        :param bert: BERT model which should be trained
        :param token_vocab_size: total token vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.MLM = MaskedLanguageModel(self.bert.hidden, token_vocab_size)
        self.eMLM = MaskedLanguageModel(self.bert.hidden, token_vocab_size)

    def forward(self, x, entity_masked_x=None, itype_seq=None, opnd_type_seq=None, reg_id_seq=None, reg_r_seq=None, reg_w_seq=None, eflags_seq=None):
        x = self.bert.forward(x, itype_seq, opnd_type_seq, reg_id_seq, reg_r_seq, reg_w_seq, eflags_seq)
        

        if entity_masked_x is not None:
            entity_masked_x = self.bert.forward(entity_masked_x, itype_seq, opnd_type_seq, reg_id_seq, reg_r_seq, reg_w_seq, eflags_seq)
            return self.MLM(x), self.eMLM(entity_masked_x)
        else:
            return self.MLM(x), None


class NextSentencePrediction(nn.Module):
    """
    From NSP task, now used for DUP and CWP
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = token_vocab_size
    """

    def __init__(self, hidden, token_vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, token_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


