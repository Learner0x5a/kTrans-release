import torch
import random
import AsmLM
import numpy as np
from tqdm import tqdm
from glob import glob
from torch.optim import Adam, AdamW
from lightning import LightningModule
from config import TOKENIZER, VOCAB_SIZES, SEED
from transformers import get_linear_schedule_with_warmup

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.multiprocessing.set_sharing_strategy('file_system')

class AsmLMModule(LightningModule):
    def __init__(self, total_steps, warmup_steps_ratio=0.01, lr: float = 3e-5, betas=(0.9, 0.999), weight_decay: float = 0.01):
        super().__init__()
        self.total_steps = total_steps
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_steps_ratio = warmup_steps_ratio
        self.tokenizer = TOKENIZER
        self.bert = AsmLM.model.BERT(VOCAB_SIZES, hidden=768, n_layers=12, attn_heads=8, dropout=0.0)
        self.model = AsmLM.model.BERTLM(self.bert, VOCAB_SIZES['token_vocab'])

        self.masked_criterion = torch.nn.NLLLoss(ignore_index=0)
        self.entity_masked_criterion = torch.nn.NLLLoss(ignore_index=0)

        self.train_losses = []
        self.train_mlm_losses = []
        self.train_entity_mlm_losses = []
        self.eval_losses = []
        self.eval_mlm_losses = []
        self.eval_entity_mlm_losses = []

    def forward(self, x, entity_masked_x=None, itype_seq=None, opnd_type_seq=None, reg_id_seq=None, reg_r_seq=None, reg_w_seq=None, eflags_seq=None):
        return self.model.forward(x, entity_masked_x, itype_seq, opnd_type_seq, reg_id_seq, reg_r_seq, reg_w_seq, eflags_seq)

    def training_step(self, batch, batch_idx):
        results = {}
        
        mlm_output, entity_mlm_output = self.forward(
            batch['masked_func_token_ids'],
            batch['entity_masked_func_token_ids'], 
            batch['func_insn_type_ids'], 
            batch['func_opnd_type_ids'], 
            batch['func_reg_id_ids'], 
            batch['func_opnd_r_ids'], 
            batch['func_opnd_w_ids'], 
            batch['func_eflags_ids'], 
        )

        mlm_loss = self.masked_criterion(mlm_output.transpose(1, 2), batch['masked_func_token_labels'])
        if entity_mlm_output is not None:
            entity_mlm_loss = self.entity_masked_criterion(entity_mlm_output.transpose(1, 2), batch['entity_masked_func_token_labels'])
        else:
            entity_mlm_loss = 0.0*mlm_loss # use mlm_loss to stay on the same device
        loss = mlm_loss + entity_mlm_loss

        results['loss'] = loss
        results['mlm_loss'] = mlm_loss.item()
        results['entity_mlm_loss'] = entity_mlm_loss.item()

        self.train_losses.append(loss.item())

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log("train_loss_mask", mlm_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log("train_loss_entity_mask", entity_mlm_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

        self.log("lr", self.lr_schedulers().get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

        return results


    def on_training_epoch_end(self):
        self.log("train_mean_loss", np.mean(self.train_losses), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)        

    def validation_step(self, batch, batch_idx):
        results = {}

        mlm_output, entity_mlm_output = self.forward(
            batch['masked_func_token_ids'],
            batch['entity_masked_func_token_ids'], 
            batch['func_insn_type_ids'], 
            batch['func_opnd_type_ids'], 
            batch['func_reg_id_ids'], 
            batch['func_opnd_r_ids'], 
            batch['func_opnd_w_ids'], 
            batch['func_eflags_ids'], 
        )

        mlm_loss = self.masked_criterion(mlm_output.transpose(1, 2), batch['masked_func_token_labels'])
        if entity_mlm_output is not None:
            entity_mlm_loss = self.entity_masked_criterion(entity_mlm_output.transpose(1, 2), batch['entity_masked_func_token_labels'])
        else:
            entity_mlm_loss = 0.0*mlm_loss # use mlm_loss to stay on the same device
        loss = mlm_loss + entity_mlm_loss

        results['loss'] = loss
        results['mlm_loss'] = mlm_loss.item()
        results['entity_mlm_loss'] = entity_mlm_loss.item()

        self.eval_losses.append(loss.item())
        self.eval_mlm_losses.append(mlm_loss.item())
        self.eval_entity_mlm_losses.append(entity_mlm_loss.item())

        self.log("eval_loss", loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log("eval_loss_mask", mlm_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log("eval_loss_entity_mask", entity_mlm_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        return results

    def on_validation_epoch_end(self):
        mean_mask_loss = np.mean(self.eval_mlm_losses)
        mean_entity_mask_loss = np.mean(self.eval_entity_mlm_losses)
        perplexity = np.exp(mean_mask_loss) + np.exp(mean_entity_mask_loss)

        self.log("eval_mean_loss", np.mean(self.eval_losses), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("eval_mean_mask_loss", mean_mask_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("eval_mean_entity_mask_loss", mean_entity_mask_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("eval_perplexity", perplexity, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        results = {}

        embs = self.bert.encode(
            batch['func_token_ids'],
            batch['func_insn_type_ids'], 
            batch['func_opnd_type_ids'],
            batch['func_reg_id_ids'],
            batch['func_opnd_r_ids'], 
            batch['func_opnd_w_ids'],
            batch['func_eflags_ids'],
        )

        results['embeddings'] = embs

        return results

    def configure_optimizers(self):
        optim = AdamW(self.model.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        optim_schedule = get_linear_schedule_with_warmup(optim, int(self.warmup_steps_ratio*self.total_steps), self.total_steps)

        return [optim], [{"scheduler": optim_schedule, "interval": "step"}]
