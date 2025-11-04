import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from Models.Bert.Bert import Bert
from Models.Layers import (
    MaxPooling, CNN, dropout, RNN_from_opt, set_dropout_prob, weighted_avg,
    set_seq_dropout, Attention, DeepAttention, LinearSelfAttn, GetFinalScores
)
from Utils.CoQAUtils import POS, ENT


class SDNet(nn.Module):
    def __init__(self, opt, word_embedding):
        super(SDNet, self).__init__()
        print('SDNet model\n')

        self.opt = opt
        self.drop_emb = False
        self.use_cuda = bool(self.opt.get('cuda', torch.cuda.is_available()))
        set_dropout_prob(0.0 if 'DROPOUT' not in opt else float(opt['DROPOUT']))
        set_seq_dropout('VARIATIONAL_DROPOUT' in self.opt)

        x_input_size = 0
        ques_input_size = 0

        # --- word embeddings ---
        self.vocab_size = int(opt['vocab_size'])
        vocab_dim = int(opt['vocab_dim'])
        self.vocab_embed = nn.Embedding(self.vocab_size, vocab_dim, padding_idx=1)
        self.vocab_embed.weight.data = word_embedding
        x_input_size += vocab_dim
        ques_input_size += vocab_dim

        # --- character CNN (optional) ---
        if 'CHAR_CNN' in self.opt:
            print('CHAR_CNN')
            char_vocab_size = int(opt['char_vocab_size'])
            char_dim = int(opt['char_emb_size'])
            char_hidden_size = int(opt['char_hidden_size'])
            self.char_embed = nn.Embedding(char_vocab_size, char_dim, padding_idx=1)
            self.char_cnn = CNN(char_dim, 3, char_hidden_size)
            self.maxpooling = MaxPooling()
            x_input_size += char_hidden_size
            ques_input_size += char_hidden_size

        if 'TUNE_PARTIAL' in self.opt:
            print('TUNE_PARTIAL')
            self.fixed_embedding = word_embedding[opt['tune_partial']:]
        else:
            self.vocab_embed.weight.requires_grad = False

        # --- contextual (BERT / LoRA) ---
        cdim = 0
        self.use_contextual = False
        if 'BERT' in self.opt:
            print('Using BERT')
            self.Bert = Bert(self.opt)

            if 'LOCK_BERT' in self.opt and 'BERT_LORA' not in self.opt:
                print("Lock BERT's weights (no LoRA)")
                for p in self.Bert.parameters():
                    p.requires_grad = False
            elif 'BERT_LORA' in self.opt:
                print('LoRA enabled: base freezing handled inside Bert wrapper')

            if 'BERT_LARGE' in self.opt:
                bert_dim = 1024
            else:
                bert_dim = 768
            print('BERT dim:', bert_dim)

            self.use_contextual = True
            cdim = bert_dim
            x_input_size += bert_dim
            ques_input_size += bert_dim

        # --- lexical pre-align attention (GloVe-level) ---
        self.pre_align = Attention(vocab_dim, opt['prealign_hidden'], correlation_func=3, do_similarity=True)
        x_input_size += vocab_dim  # pre-align concat

        # --- POS/NER/features ---
        pos_dim = opt['pos_dim']
        ent_dim = opt['ent_dim']
        self.pos_embedding = nn.Embedding(len(POS), pos_dim)
        self.ent_embedding = nn.Embedding(len(ENT), ent_dim)

        x_feat_len = 4
        if 'ANSWER_SPAN_IN_CONTEXT_FEATURE' in self.opt:
            print('ANSWER_SPAN_IN_CONTEXT_FEATURE')
            x_feat_len += 1
        x_input_size += pos_dim + ent_dim + x_feat_len

        print('Initially, the vector_sizes [doc, query] are', x_input_size, ques_input_size)

        addtional_feat = cdim if self.use_contextual else 0

        # --- input RNN encoders ---
        self.context_rnn, context_rnn_output_size = RNN_from_opt(
            x_input_size, opt['hidden_size'],
            num_layers=opt['in_rnn_layers'], concat_rnn=opt['concat_rnn'],
            add_feat=addtional_feat
        )
        self.ques_rnn, ques_rnn_output_size = RNN_from_opt(
            ques_input_size, opt['hidden_size'],
            num_layers=opt['in_rnn_layers'], concat_rnn=opt['concat_rnn'],
            add_feat=addtional_feat
        )
        print('After Input LSTM, the vector_sizes [doc, query] are [',
              context_rnn_output_size, ques_rnn_output_size, '] *', opt['in_rnn_layers'])

        # --- deep inter-attention (multi-level) ---
        self.deep_attn = DeepAttention(
            opt,
            abstr_list_cnt=opt['in_rnn_layers'],
            deep_att_hidden_size_per_abstr=opt['deep_att_hidden_size_per_abstr'],
            correlation_func=3,
            word_hidden_size=vocab_dim + addtional_feat  # allow BERT+GloVe at word-level
        )
        self.deep_attn_input_size = self.deep_attn.rnn_input_size
        self.deep_attn_output_size = self.deep_attn.output_size  # typically 250

        # --- question understanding ---
        self.high_lvl_ques_rnn, high_lvl_ques_rnn_output_size = RNN_from_opt(
            ques_rnn_output_size * opt['in_rnn_layers'],
            opt['highlvl_hidden_size'], num_layers=opt['question_high_lvl_rnn_layers'],
            concat_rnn=True
        )

        # --- self-attention over context ---
        # Input to self-attn = [inter_attn_out (250) | inter_attn_features | (optional) BERT | GloVe]
        self.after_deep_attn_size = (
            self.deep_attn_output_size + self.deep_attn_input_size + addtional_feat + vocab_dim
        )
        self.self_attn_input_size = self.after_deep_attn_size

        # Attention returns vectors with SAME width as inter-attn output (250) in this repo's implementation.
        self.highlvl_self_att = Attention(
            self.self_attn_input_size, opt['deep_att_hidden_size_per_abstr'], correlation_func=3
        )
        print('Self deep-attention input is {}-dim'.format(self.self_attn_input_size))

        # ROBUST FIX:
        # self-attn output is already 250 in this codebase; keep an identity "proj" for clarity,
        # and SUM with inter-attn output so the HL LSTM always sees 250 dims.
        self.self_attn_out_proj = nn.Identity()

        self.high_lvl_context_rnn, high_lvl_context_rnn_output_size = RNN_from_opt(
            self.deep_attn_output_size,  # expect 250-d per token
            opt['highlvl_hidden_size'], num_layers=1, concat_rnn=False
        )
        context_final_size = high_lvl_context_rnn_output_size

        print('Do Question self attention')
        self.ques_self_attn = Attention(
            high_lvl_ques_rnn_output_size, opt['query_self_attn_hidden_size'], correlation_func=3
        )

        ques_final_size = high_lvl_ques_rnn_output_size
        print('Before answer span finding, hidden size are', context_final_size, ques_final_size)

        # --- merge question and predict ---
        self.ques_merger = LinearSelfAttn(ques_final_size)
        self.get_answer = GetFinalScores(context_final_size, ques_final_size)

    def forward(
        self, x, x_single_mask, x_char, x_char_mask, x_features, x_pos, x_ent,
        x_bert, x_bert_mask, x_bert_offsets,
        q, q_mask, q_char, q_char_mask,
        q_bert, q_bert_mask, q_bert_offsets,
        context_len
    ):
        batch_size = q.shape[0]
        x_mask = x_single_mask.expand(batch_size, -1)
        x_word_embed = self.vocab_embed(x).expand(batch_size, -1, -1)       # [B, Lx, vocab_dim]
        ques_word_embed = self.vocab_embed(q)                                # [B, Lq, vocab_dim]

        x_input_list = [dropout(x_word_embed, p=self.opt['dropout_emb'], training=self.drop_emb)]
        ques_input_list = [dropout(ques_word_embed, p=self.opt['dropout_emb'], training=self.drop_emb)]

        # --- contextual embeddings (BERT) ---
        x_cemb = ques_cemb = None
        if 'BERT' in self.opt:
            x_cemb_mid = self.Bert(x_bert, x_bert_mask, x_bert_offsets, x_single_mask)  # [1, Lx, cdim]
            x_cemb_mid = x_cemb_mid.expand(batch_size, -1, -1)
            ques_cemb_mid = self.Bert(q_bert, q_bert_mask, q_bert_offsets, q_mask)      # [B, Lq, cdim]

            x_cemb = x_cemb_mid
            ques_cemb = ques_cemb_mid

            x_input_list.append(x_cemb_mid)
            ques_input_list.append(ques_cemb_mid)

        # --- char CNN (optional) ---
        if 'CHAR_CNN' in self.opt:
            x_char_final = self.character_cnn(x_char, x_char_mask).expand(batch_size, -1, -1)
            ques_char_final = self.character_cnn(q_char, q_char_mask)
            x_input_list.append(x_char_final)
            ques_input_list.append(ques_char_final)

        # --- lexical pre-align features ---
        x_prealign = self.pre_align(x_word_embed, ques_word_embed, q_mask)
        x_input_list.append(x_prealign)

        # --- POS/NER/features ---
        x_pos_emb = self.pos_embedding(x_pos).expand(batch_size, -1, -1)
        x_ent_emb = self.ent_embedding(x_ent).expand(batch_size, -1, -1)
        x_input_list += [x_pos_emb, x_ent_emb, x_features]

        # concat inputs
        x_input = torch.cat(x_input_list, 2)        # [B, Lx, ...]
        ques_input = torch.cat(ques_input_list, 2)  # [B, Lq, ...]

        # --- BiLSTM encoders ---
        _, x_rnn_layers = self.context_rnn(x_input, x_mask, return_list=True, x_additional=x_cemb)
        _, ques_rnn_layers = self.ques_rnn(ques_input, q_mask, return_list=True, x_additional=ques_cemb)

        # --- question high-level ---
        ques_highlvl = self.high_lvl_ques_rnn(torch.cat(ques_rnn_layers, 2), q_mask)
        ques_rnn_layers.append(ques_highlvl)

        # --- deep inter-attention ---
        if x_cemb is None:
            x_long = x_word_embed
            ques_long = ques_word_embed
        else:
            x_long = torch.cat([x_word_embed, x_cemb], 2)
            ques_long = torch.cat([ques_word_embed, ques_cemb], 2)

        x_rnn_after_inter_attn, x_inter_attn = self.deep_attn(
            [x_long], x_rnn_layers, [ques_long], ques_rnn_layers, x_mask, q_mask, return_bef_rnn=True
        )
        # x_rnn_after_inter_attn: [B, Lx, deep_attn_output_size]  (≈250)
        # x_inter_attn:           [B, Lx, deep_attn_input_size]

        # --- deep self-attention over context ---
        if x_cemb is None:
            x_self_attn_input = torch.cat([x_rnn_after_inter_attn, x_inter_attn, x_word_embed], 2)
        else:
            x_self_attn_input = torch.cat([x_rnn_after_inter_attn, x_inter_attn, x_cemb, x_word_embed], 2)
        # x_self_attn_input width == self.self_attn_input_size

        x_self_attn_output = self.highlvl_self_att(
            x_self_attn_input, x_self_attn_input, x_mask, x3=x_rnn_after_inter_attn, drop_diagonal=True
        )
        # In this codebase Attention returns 250-d; keep identity for clarity
        x_self_attn_output = self.self_attn_out_proj(x_self_attn_output)  # [B, Lx, 250]

        # robust fix: SUM (not concat) so the LSTM sees exactly 250 dims
        hlc_input = x_rnn_after_inter_attn + x_self_attn_output           # [B, Lx, 250]
        x_highlvl_output = self.high_lvl_context_rnn(hlc_input, x_mask)
        x_final = x_highlvl_output  # [B, Lx, context_final_size]

        # --- question self-attention & merge ---
        ques_final = self.ques_self_attn(ques_highlvl, ques_highlvl, q_mask, x3=None, drop_diagonal=True)
        q_merge_weights = self.ques_merger(ques_final, q_mask)
        ques_merged = weighted_avg(ques_final, q_merge_weights)  # [B, ques_final_size]

        # --- predict ---
        # NOTE: GetFinalScores returns (score_s, score_e, score_no, score_yes, score_noanswer)
        score_s, score_e, score_no, score_yes, score_noanswer = self.get_answer(x_final, ques_merged, x_mask)
        # Trainer expects order: (score_s, score_e, score_yes, score_no, score_no_answer)
        return score_s, score_e, score_yes, score_no, score_noanswer

    def character_cnn(self, x_char, x_char_mask):
        x_char_embed = self.char_embed(x_char)  # [B, W, C, D]
        batch_size = x_char_embed.shape[0]
        word_num = x_char_embed.shape[1]
        char_num = x_char_embed.shape[2]
        char_dim = x_char_embed.shape[3]
        x_char_cnn = self.char_cnn(
            x_char_embed.contiguous().view(-1, char_num, char_dim), x_char_mask
        )  # (B*W) x C x H
        x_char_cnn_final = self.maxpooling(
            x_char_cnn, x_char_mask.contiguous().view(-1, char_num)
        ).contiguous().view(batch_size, word_num, -1)  # [B, W, H]
        return x_char_cnn_final
