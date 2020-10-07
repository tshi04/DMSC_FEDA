'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from LeafNATS.data.utils import create_batch_memory
from LeafNATS.eval_scripts.utils import eval_accu_mse_v1
from LeafNATS.utils.utils import show_progress

from .attn_vis import createHTML
from .model import modelDMSC


class modelView(modelDMSC):

    def __init__(self, args):
        super().__init__(args=args)

    def visualization(self):
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

        vis_dir = '../nats_results/attn_vis'
        if not os.path.exists(vis_dir):
            os.mkdir(vis_dir)
        else:
            shutil.rmtree(vis_dir)
            os.mkdir(vis_dir)

        for model_name in self.base_models:
            self.base_models[model_name].eval()
        for model_name in self.train_models:
            self.train_models[model_name].eval()
        with torch.no_grad():

            print('Begin Visualization')
            n_batch = len(self.vis_data)
            print('The number of batches (visualization): {}'.format(n_batch))
            for batch_id in range(n_batch):

                self.build_batch(self.vis_data[batch_id])
                self.visualization_worker(batch_id, vis_dir)

                show_progress(batch_id+1, n_batch)
            print()

    def visualization_worker(self, batch_id, vis_dir):
        '''
        For visualization attention weights.
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

        logits = []
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
            cv_hidden1 = torch.bmm(attn1.unsqueeze(1), review_enc).squeeze(1)
            attn1_out.append(0.5*attn0 + 0.5*attn1)

            cv_hidden = cv_hidden0 + cv_hidden1
            fc = torch.relu(self.train_models['drop'](
                self.train_models['ff'][k](cv_hidden)))
            logits.append(self.train_models['drop'](
                self.train_models['classifier'][k](fc)))

        logits = torch.cat(logits, 0)
        logits = logits.view(self.args.n_tasks, batch_size, self.args.n_class)
        logits = logits.transpose(0, 1)

        logits = self.build_pipe()
        logits = torch.softmax(logits, dim=2)

        ratePred = logits.topk(1, dim=2)[1].squeeze(2).data.cpu().numpy()
        ratePred += 1
        ratePred = ratePred.tolist()

        rateTrue = self.batch_data['rating'].data.cpu().numpy()
        rateTrue += 1
        rateTrue = rateTrue.tolist()

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
            text = [' '.join(review[k])]*self.args.n_tasks
            weights1 = []
            weights2 = []
            for j in range(self.args.n_tasks):
                weights1.append(attn0_out[j][k])
                weights2.append(attn1_out[j][k])
            barate = rateTrue[k]
            pdrate = ratePred[k]
            _, mse = eval_accu_mse_v1(pdrate, barate)
            if len(review[k]) < 150:
                filename = os.path.join(
                    vis_dir,
                    str(batch_id) + '_' + str(k) +
                    '_' + str(np.average(barate))
                    + '_' + str(mse) + '_' + str(len(review[k])) + '.html')
                createHTML(text, weights1, weights2, barate, pdrate, filename)
