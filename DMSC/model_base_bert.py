'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import os
import time

import numpy as np
import torch
from torch.autograd import Variable

from .data.process_minibatch_v3 import process_minibatch
from .model_base import modelDMSCBase


class modelDMSCBaseBert(modelDMSCBase):
    '''
    Classfication.
    Rewrite vocabulary module. 
    '''

    def __init__(self, args):
        super().__init__(args=args)
        self.pretrained_models = {}

    def build_vocabulary(self):
        '''
        vocabulary
        '''
        return

    def build_batch(self, batch_):
        '''
        get batch data
        '''
        review, rating, seg, pad_mask, att_mask = process_minibatch(
            input_=batch_,
            max_lens=self.args.review_max_lens
        )
        self.batch_data['input_ids'] = review.to(self.args.device)
        self.batch_data['seg'] = seg.to(self.args.device)
        self.batch_data['rating'] = rating.to(self.args.device)
        self.batch_data['pad_mask'] = pad_mask.to(self.args.device)
        self.batch_data['att_mask'] = att_mask.to(self.args.device)
