'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import json
import re

import torch
from torch.autograd import Variable


def process_minibatch(input_, max_lens):
    '''
    Process the minibatch
    '''
    len_review = 0
    review_arr = []
    rating_arr = []
    for line in input_:
        itm = json.loads(line)

        tmp_rate = itm['label'][1:]
        rating_arr.append([int(float(rt)) for rt in tmp_rate])

        review2id = itm['bert_id']
        review_arr.append(review2id)

        if len(review2id) > len_review:
            len_review = len(review2id)

    review_lens = min(max_lens, len_review)

    review_arr = [itm[:review_lens-2] for itm in review_arr]
    review_arr = [[101] + itm + [102] + [0 for _ in range(review_lens-2-len(itm))]
                  for itm in review_arr]
    seg_arr = [[0 for _ in range(review_lens)] for k in range(len(review_arr))]

    review_var = Variable(torch.LongTensor(review_arr))
    rating_var = Variable(torch.LongTensor(rating_arr))
    rating_var -= 1
    rating_var[rating_var < 0] = -1
    seg_var = Variable(torch.LongTensor(seg_arr))

    pad_mask = Variable(torch.FloatTensor(review_arr))
    pad_mask[pad_mask != float(0)] = -1.0
    pad_mask[pad_mask == float(0)] = 0.0
    pad_mask = -pad_mask

    att_mask = Variable(torch.FloatTensor(review_arr))
    att_mask[att_mask == float(101)] = 0.0
    att_mask[att_mask == float(102)] = 0.0
    att_mask[att_mask != float(0)] = -1.0
    att_mask[att_mask == float(0)] = 0.0
    att_mask = -att_mask

    return review_var, rating_var, seg_var, pad_mask, att_mask
