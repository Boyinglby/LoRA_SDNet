# --- add these imports at the top ---
from peft import LoraConfig, get_peft_model, TaskType
import os
from typing import Optional, List

import torch
import torch.nn as nn

from transformers import AutoModel, AutoConfig
from transformers import BertModel


class Bert(nn.Module):
    def __init__(self, opt):
        super(Bert, self).__init__()
        print("Loading BERT model...")

        self.BERT_MAX_LEN = 512

        # We are removing weighted layer mix, so ignore the old flag:
        self.linear_combine = False  # force off

       # Decide which model to load: prefer local path if it exists; otherwise fallback to HF model name
        is_large = "BERT_LARGE" in opt
        if is_large:
            print("Using BERT Large model")
            # Original code used opt['datadir'] + opt['BERT_large_model_file'] (a local folder or archive)
            # We try local first; if missing, use HF hub "bert-large-uncased"
            local_path = os.path.join(opt.get("datadir", ""), opt.get("BERT_large_model_file", ""))
            hf_name = opt.get("BERT_large_hf_name", "bert-large-uncased")
        else:
            print("Using BERT base model")
            local_path = os.path.join(opt.get("datadir", ""), opt.get("BERT_model_file", ""))
            # Original comment shows 'bert-base-cased'; keep that as default
            hf_name = opt.get("BERT_base_hf_name", "bert-base-cased")

        model_src = local_path if local_path and os.path.exists(local_path) else hf_name
        if model_src == local_path:
            print("Loading BERT model from (local):", model_src)
        else:
            print("Loading BERT model from HF hub:", model_src)

        config = AutoConfig.from_pretrained(model_src, output_hidden_states=True)
        self.bert_model = AutoModel.from_pretrained(model_src, config=config)

        # >>> NEW: attach LoRA if requested <<<
        if "BERT_LORA" in opt:
            print("Attaching LoRA adapters to BERT")
            # Freeze base model
            for p in self.bert_model.parameters():
                p.requires_grad = False

            # Typical LoRA config for BERT
            r = int(opt.get("LORA_R", 8))
            alpha = int(opt.get("LORA_ALPHA", 16))
            lora_dropout = float(opt.get("LORA_DROPOUT", 0.05))

            # target module names for HF BERT:
            # these substrings match modules like:
            #   attention.self.query/key/value, attention.output.dense,
            #   intermediate.dense, output.dense
            target_modules = opt.get(
                "LORA_TARGETS",
                ["query", "key", "value", "dense"]
            )

            lconf = LoraConfig(
                r=int(opt.get("LORA_R", 8)),
                lora_alpha=int(opt.get("LORA_ALPHA", 16)),
                lora_dropout=float(opt.get("LORA_DROPOUT", 0.05)),
                bias="none",
                target_modules=opt.get("LORA_TARGETS", ["query","key","value","dense"]),
                task_type=TaskType.FEATURE_EXTRACTION,   # <<< THIS is the key change
            )
            self.bert_model = get_peft_model(self.bert_model, lconf)
            # LoRA params now have requires_grad=True

        self.bert_dim = config.hidden_size
        self.bert_layer = config.num_hidden_layers

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model.to(self.device)

        # IMPORTANT: don't force eval(); SDNet will call .train()/.eval() globally
        print("Finished loading")

    def forward(self, x_bert, x_bert_mask, x_bert_offset, x_mask):
        # --- collect LAST hidden states in 512-chunks (no no_grad if LoRA is on) ---
        bert_sent_len = x_bert.shape[1]
        last_layers = []
        p = 0
        while p < bert_sent_len:
            input_ids = x_bert[:, p:(p + self.BERT_MAX_LEN)].to(self.device)
            attn_mask = x_bert_mask[:, p:(p + self.BERT_MAX_LEN)].to(self.device)

            # Allow grads so LoRA trains; base weights are frozen anyway
            out = self.bert_model(input_ids=input_ids, attention_mask=attn_mask)
            last_layers.append(out.last_hidden_state)
            p += self.BERT_MAX_LEN

        bert_embedding = torch.cat(last_layers, dim=1)  # [B, full_seq_len, H]

        # Map subtokens -> word-level by averaging [st:ed)
        batch_size = x_mask.shape[0]
        max_word_num = x_mask.shape[1]
        output = torch.zeros(batch_size, max_word_num, self.bert_dim, device=self.device)

        for i in range(batch_size):
            for j in range(max_word_num):
                if x_mask[i, j] == 0:
                    continue
                st = int(x_bert_offset[i, j, 0]); ed = int(x_bert_offset[i, j, 1])
                if st < ed:
                    if st + 1 == ed:
                        output[i, j, :] = bert_embedding[i, st, :]
                    else:
                        output[i, j, :] = bert_embedding[i, st:ed, :].mean(dim=0)

        # Return a **trainable** tensor (do NOT detach), so LoRA grads flow
        return output