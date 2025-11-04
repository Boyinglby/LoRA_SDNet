# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Optional, List

import torch
import torch.nn as nn

from transformers import AutoModel, AutoConfig
from transformers import BertModel


class Bert(nn.Module):
    """
    Modernized BERT wrapper using 🤗 Transformers.
    Keeps the same public API as the original:
      - __init__(opt)
      - forward(x_bert, x_bert_mask, x_bert_offset, x_mask)
    Also supports the "BERT_LINEAR_COMBINE" option in opt.
    """
    def __init__(self, opt):
        super(Bert, self).__init__()
        print("Loading BERT model...")

        self.BERT_MAX_LEN = 512
        self.linear_combine = "BERT_LINEAR_COMBINE" in opt

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

        # Ask Transformers to return all hidden states (embeddings + each encoder layer)
        config = AutoConfig.from_pretrained(model_src, output_hidden_states=True)
        self.bert_model = AutoModel.from_pretrained(model_src, config=config)

        # Dimensions
        self.bert_dim: int = config.hidden_size
        # Note: hidden_states length = num_hidden_layers + 1 (embeddings)
        self.bert_layer: int = config.num_hidden_layers

        # Device handling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model.to(self.device)
        self.bert_model.eval()

        print("Finished loading")

    """
        Input:
          x_bert:        (batch, max_bert_sent_len) token ids (already numericized)
          x_bert_mask:   (batch, max_bert_sent_len) attention mask (0/1)
          x_bert_offset: (batch, max_real_word_num, 2) start/end offsets into subword sequence
          x_mask:        (batch, max_real_word_num) (0/1) which real words are valid

        Output:
          If not linear_combine:
            embedding: (batch, max_real_word_num, bert_dim)   # last-layer word-level average
          If linear_combine:
            List[Tensor] of length = self.bert_layer
            each tensor: (batch, max_real_word_num, bert_dim) # per-layer word-level average
    """

    def forward(self, x_bert, x_bert_mask, x_bert_offset, x_mask):
        if self.linear_combine:
            return self._combine_forward(x_bert, x_bert_mask, x_bert_offset, x_mask)

        # --- Collect last hidden states in 512-chunks (to mimic the original logic) ---
        bert_sent_len = x_bert.shape[1]
        last_layers: List[torch.Tensor] = []
        p = 0
        while p < bert_sent_len:
            input_ids = x_bert[:, p:(p + self.BERT_MAX_LEN)].to(self.device)
            attn_mask = x_bert_mask[:, p:(p + self.BERT_MAX_LEN)].to(self.device)

            with torch.no_grad():
                out = self.bert_model(input_ids=input_ids, attention_mask=attn_mask)
            # out.last_hidden_state: (batch, chunk_len, hidden)
            last_layers.append(out.last_hidden_state)

            p += self.BERT_MAX_LEN

        # (batch, full_seq_len, hidden)
        bert_embedding = torch.cat(last_layers, dim=1)

        # --- Map subword sequence to word-level with simple average over [st:ed) ---
        batch_size = x_mask.shape[0]
        max_word_num = x_mask.shape[1]

        output = torch.zeros(batch_size, max_word_num, self.bert_dim, device=self.device)
        # ensure offsets & masks are on CPU/Tensor for indexing, then use .item() to avoid tensors in loops
        x_mask_local = x_mask
        x_bert_offset_local = x_bert_offset

        for i in range(batch_size):
            for j in range(max_word_num):
                if x_mask_local[i, j] == 0:
                    continue
                st = int(x_bert_offset_local[i, j, 0])
                ed = int(x_bert_offset_local[i, j, 1])
                if st + 1 == ed:
                    output[i, j, :] = bert_embedding[i, st, :]
                else:
                    if st < ed:
                        # average the subword span
                        output[i, j, :] = bert_embedding[i, st:ed, :].mean(dim=0)

        return output.detach


    def _combine_forward(self, x_bert, x_bert_mask, x_bert_offset, x_mask):
        """
        Return per-layer word-level embeddings (list of length self.bert_layer).
        Equivalent to the old implementation's behavior, but using Transformers.
        """
        bert_sent_len = x_bert.shape[1]
        # We'll gather hidden_states from all chunks and then stitch them
        # Note: hidden_states[0] is embeddings; [1..L] are encoder layers
        layer_buffers: List[List[torch.Tensor]] = [[] for _ in range(self.bert_layer + 1)]

        p = 0
        while p < bert_sent_len:
            input_ids = x_bert[:, p:(p + self.BERT_MAX_LEN)].to(self.device)
            attn_mask = x_bert_mask[:, p:(p + self.BERT_MAX_LEN)].to(self.device)
            with torch.no_grad():
                out = self.bert_model(input_ids=input_ids, attention_mask=attn_mask)
            # hidden_states: tuple(len=L+1) of (batch, chunk_len, hidden)
            hidden_states = out.hidden_states  # type: ignore

            for li, h in enumerate(hidden_states):
                layer_buffers[li].append(h)

            p += self.BERT_MAX_LEN

        # Concatenate chunks along sequence axis for each layer
        # layer 0: embeddings; layers 1..L: encoder layers
        full_layers = [torch.cat(bufs, dim=1) for bufs in layer_buffers]  # each (batch, full_seq_len, hidden)

        batch_size = x_mask.shape[0]
        max_word_num = x_mask.shape[1]

        # For parity with the original `combine_forward`, we return only the encoder layers (exclude embeddings)
        outputs: List[torch.Tensor] = []
        for Lidx in range(1, self.bert_layer + 1):
            layer_tensor = full_layers[Lidx]
            word_level = torch.zeros(batch_size, max_word_num, self.bert_dim, device=self.device)

            for i in range(batch_size):
                for j in range(max_word_num):
                    if x_mask[i, j] == 0:
                        continue
                    st = int(x_bert_offset[i, j, 0])
                    ed = int(x_bert_offset[i, j, 1])
                    if st + 1 == ed:
                        word_level[i, j, :] = layer_tensor[i, st, :]
                    else:
                        if st < ed:
                            word_level[i, j, :] = layer_tensor[i, st:ed, :].mean(dim=0)

            outputs.append(word_level.detach())

        return outputs
