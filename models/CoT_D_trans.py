# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : CoT_Medicator.py
# @Time         : Created at 2020/4/20
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn.functional as F

from models.generator import LSTMGenerator, TranGenerator


class Cot_D_trans(TranGenerator):
    def __init__(self, embedding_dim, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(Cot_D_trans, self).__init__(embedding_dim, vocab_size, max_seq_len, padding_idx, gpu)

    def get_pred(self, input, target):
        pred = self.forward(input)
        target_onehot = F.one_hot(target.view(-1), self.vocab_size).float()
        pred = torch.sum(pred * target_onehot, dim=-1)
        return pred
