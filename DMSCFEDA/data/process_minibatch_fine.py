'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import json
import re
from pprint import pprint

import numpy as np
import torch
from torch.autograd import Variable


def process_minibatch(input_, vocab2id, max_lens):
    '''
    Process the minibatch for beeradvocate and tripadvisor datasets
    The data format
    {id: [], aspect_label: [], sentiment_label: [], text: [], keywords: []}
    '''
    len_review = 0
    review_arr = []
    sent_arr = []
    aspect_arr = []
    for line in input_:
        itm = json.loads(line)

        aspect_arr.append(itm['aspect_label'])
        sent_arr.append(itm['sentiment_label'])

        review = itm['text'].split()
        review2id = [vocab2id[wd] if wd in vocab2id else vocab2id['<unk>']
                     for wd in review]
        review_arr.append(review2id)

        if len_review < len(review2id):
            len_review = len(review2id)

    review_lens = min(max_lens, len_review)

    review_arr = [itm[:review_lens] for itm in review_arr]
    review_arr = [itm + [vocab2id['<pad>'] for _ in range(review_lens-len(itm))]
                  for itm in review_arr]
    review_var = Variable(torch.LongTensor(review_arr))

    pad_mask = Variable(torch.FloatTensor(review_arr))
    pad_mask[pad_mask != float(vocab2id['<pad>'])] = -1.0
    pad_mask[pad_mask == float(vocab2id['<pad>'])] = 0.0
    pad_mask = -pad_mask

    aspect_var = Variable(torch.LongTensor(aspect_arr))
    aspect_var[aspect_var < 0] = -1

    sent_var = Variable(torch.LongTensor(sent_arr))
    sent_var -= 1
    sent_var[sent_var < 0] = -1

    return review_var, pad_mask, aspect_var, sent_var


def process_minibatch_test(input_, vocab2id, max_lens):
    '''
    For testing only.
    '''
    rate_arr = []
    review_arr = []
    review_text = []
    review_ids = []
    len_sen = 0
    for itm in input_:
        arr = re.split(r'\t\t\t', itm[:-1])

        rating = arr[0].split()[1:]
        rating = [int(wd) for wd in rating]
        rate_arr.append(rating)

        review_ids.append(arr[1])
        review_text.append(re.split(r'<ssssss>', arr[2]))

        review = re.split(r'<ssssss>', arr[2])
        for k in range(len(review)):
            review[k] = review[k].split()
            review[k] = [vocab2id[wd] if wd in vocab2id else vocab2id['<unk>']
                         for wd in review[k]]
            if len_sen < len(review[k]):
                len_sen = len(review[k])
        review_arr.append(review)

    len_sen = min(len_sen, max_lens)

    review_var = []
    review_mask = []
    for k in range(len(review_arr)):
        out = [itm[:len_sen] for itm in review_arr[k]]
        out = [itm + [vocab2id['<pad>'] for _ in range(len_sen-len(itm))]
               for itm in out]
        var = Variable(torch.LongTensor(out))
        review_var.append(var)

        mask = Variable(torch.FloatTensor(out))
        mask[mask != float(vocab2id['<pad>'])] = -1.0
        mask[mask == float(vocab2id['<pad>'])] = 0.0
        mask = -mask
        review_mask.append(mask)

    rating_var = Variable(torch.LongTensor(rate_arr))
    rating_var -= 1
    rating_var[rating_var < 0] = -1

    return review_var, review_mask, rating_var, review_text, review_ids
