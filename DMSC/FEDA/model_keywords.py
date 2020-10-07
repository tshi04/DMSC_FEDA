'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import itertools
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from LeafNATS.data.utils import create_batch_memory
from LeafNATS.utils.utils import show_progress
from nltk.corpus import stopwords

from .model import modelDMSC

stop_words = stopwords.words('english')


class modelKeywords(modelDMSC):

    def __init__(self, args):
        super().__init__(args=args)

        # keywords from attention
        self.keywords0 = [{} for k in range(args.n_tasks)]
        # keywords from both attention and deliberated attention.
        self.keywords1 = [{} for k in range(args.n_tasks)]
        self.wd_freq = {}

    def keyword_extraction(self):
        '''
        Visualization
        '''
        self.build_vocabulary()
        self.build_models()
        print(self.base_models)
        print(self.train_models)
        if len(self.base_models) > 0:
            self.init_base_model_params()
        if len(self.train_models) > 0:
            self.init_train_model_params()

        self.vis_data = create_batch_memory(
            path_=self.args.data_dir,
            file_=self.args.file_vis,
            is_shuffle=False,
            batch_size=self.args.batch_size,
            is_lower=self.args.is_lower
        )

        key_dir = '../nats_results/attn_keywords'
        if not os.path.exists(key_dir):
            os.mkdir(key_dir)
        else:
            shutil.rmtree(key_dir)
            os.mkdir(key_dir)

        with torch.no_grad():

            print('Begin Generate Keywords')
            n_batch = len(self.vis_data)
            print('The number of batches (keywords): {}'.format(n_batch))
            for batch_id in range(n_batch):

                self.build_batch(self.vis_data[batch_id])
                self.keyword_worker(batch_id, key_dir)

                show_progress(batch_id+1, n_batch)
            print()

            for k in range(self.args.n_tasks):
                key_arr = [[wd, 100*self.keywords1[k][wd] /
                            (self.wd_freq[wd]+100)] for wd in self.keywords1[k]]
                key_arr = sorted(key_arr, key=lambda k: k[1])[::-1]
                key_arr = [[itm[0]]*int(round(itm[1])) for itm in key_arr
                           if (itm[0] not in stop_words) and (len(itm[0]) > 3) and (itm[0] != '<unk>')]
                key_arr = key_arr[:100]
                key_arr = list(itertools.chain(*key_arr))
                fout = open(os.path.join(key_dir, str(k)+'.txt'), 'w')
                fout.write(' '.join(key_arr) + '\n')
                fout.close()

    def keyword_worker(self, batch_id, key_dir):
        '''
        Keywords
        '''
        review_emb = self.base_models['embedding'](self.batch_data['review'])
        batch_size = review_emb.size(0)
        seq_len = review_emb.size(1)
        emb_gate = torch.sigmoid(self.train_models['gate'](review_emb))
        emb_valu = torch.relu(self.train_models['value'](review_emb))
        review_out = review_emb*(1-emb_gate) + emb_valu*emb_gate

        encoder_hy, _ = self.train_models['encoder'](review_out)

        input_pool = encoder_hy.view(batch_size, seq_len, 2, -1)
        input_pool = input_pool.contiguous().view(batch_size, seq_len*2, -1)
        max_pool = self.train_models['max_pool'](input_pool).squeeze(-1)
        max_pool = max_pool.view(batch_size, seq_len, 2)

        avg_pool = self.train_models['avg_pool'](input_pool).squeeze(-1)
        avg_pool = avg_pool.view(batch_size, seq_len, 2)

        input_fm = encoder_hy.view(batch_size, seq_len, 2, -1)

        cfmf = self.train_models['fmf'](input_fm[:, :, 0])
        cfmb = self.train_models['fmb'](input_fm[:, :, 1])
        review_enc = torch.cat((encoder_hy, max_pool, avg_pool, cfmf, cfmb), 2)

        attn0_out = []
        attn1_out = []
        for k in range(self.args.n_tasks):

            attn0 = torch.tanh(
                self.train_models['attn_forward'][k](review_enc))
            attn0 = self.train_models['attn_wrap'][k](attn0).squeeze(2)
            attn0 = torch.softmax(attn0, 1)
            cv_hidden0 = torch.bmm(attn0.unsqueeze(1), review_enc).squeeze(1)
            attn0_out.append(attn0)

            attn1 = torch.tanh(
                self.train_models['loop_forward1'][k](review_enc))
            attn1 = torch.bmm(attn1, cv_hidden0.unsqueeze(2)).squeeze(2)
            attn1 = torch.softmax(attn1, 1)
            # get the accumulated attention.
            attn1_out.append(0.5*attn0 + 0.5*attn1)

        review = []
        batch_review = self.batch_data['review'].data.cpu().numpy()
        for k in range(batch_size):
            review.append([self.batch_data['id2vocab'][wd]
                           for wd in batch_review[k] if not wd == 1])
        for k in range(self.args.n_tasks):
            attn0_out[k] = attn0_out[k].data.cpu().numpy().tolist()
            for j in range(batch_size):
                attn0_out[k][j] = attn0_out[k][j][:len(
                    review[j])]/np.sum(attn0_out[k][j][:len(review[j])])
                attn0_out[k][j] = attn0_out[k][j].tolist()
        for k in range(self.args.n_tasks):
            attn1_out[k] = attn1_out[k].data.cpu().numpy().tolist()
            for j in range(batch_size):
                attn1_out[k][j] = attn1_out[k][j][:len(
                    review[j])]/np.sum(attn1_out[k][j][:len(review[j])])
                attn1_out[k][j] = attn1_out[k][j].tolist()

        for k in range(batch_size):
            for wd in review[k]:
                try:
                    self.wd_freq[wd] += 1
                except:
                    self.wd_freq[wd] = 1
            for j in range(self.args.n_tasks):
                idx0 = np.argsort(attn0_out[j][k])[-3:]
                idx1 = np.argsort(attn1_out[j][k])[-3:]
                for id_ in idx0:
                    try:
                        self.keywords0[j][review[k][id_]] += 1
                    except:
                        self.keywords0[j][review[k][id_]] = 1
                for id_ in idx1:
                    try:
                        self.keywords1[j][review[k][id_]] += 1
                    except:
                        self.keywords1[j][review[k][id_]] = 1
