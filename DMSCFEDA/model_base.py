'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_squared_error
from torch.autograd import Variable

from .data.process_minibatch_coarse import process_minibatch
from .end2end_engine import End2EndBaseDMSCFine


class modelDMSCBase(End2EndBaseDMSCFine):
    '''
    DMSC Fine
    '''

    def __init__(self, args):
        super().__init__(args=args)

    def build_batch(self, batch_):
        '''
        get batch data
        '''
        review_txt, review, weight_mask, rating, rateOV = process_minibatch(
            input_=batch_,
            vocab2id=self.batch_data['vocab2id'],
            max_lens=self.args.review_max_lens
        )
        self.batch_data['review_txt'] = review_txt
        self.batch_data['review'] = review.to(self.args.device)
        self.batch_data['weight_mask'] = weight_mask.to(self.args.device)
        self.batch_data['rating'] = rating.to(self.args.device)
        self.batch_data['rateOV'] = rateOV.to(self.args.device)

    def build_encoder(self):
        '''
        Encoder
        '''
        raise NotImplementedError

    def build_attention(self, input_):
        '''
        Attention
        '''
        raise NotImplementedError

    def build_classifier(self, input_):
        '''
        Classifier
        '''
        raise NotImplementedError

    def build_pipe(self):
        '''
        Pipes shared by training/validation/testing
        '''
        encoder_output = self.build_encoder()
        attn_output = self.build_attention(encoder_output)
        logits_, review_vec = self.build_classifier(attn_output)

        return logits_, review_vec

    def metric_learning(self, input_):
        '''
        metric learning
        '''
        raise NotImplementedError

    def build_pipelines(self):
        '''
        Data flow from input to output.
        '''
        logits, review_vec = self.build_pipe()

        logits = logits.contiguous().view(-1, self.args.n_class)
        loss = self.loss_criterion(
            logits, self.batch_data['rating'].view(-1))

        if self.args.metric_learning:
            loss_metric = self.metric_learning(review_vec)
            loss = loss + loss_metric

        return loss

    def build_visualization(self, input_):
        '''
        visualization
        '''
        raise NotImplementedError

    def visualization_worker(self):
        '''
        visualization
        '''
        encoder_output = self.build_encoder()
        attn_output = self.build_attention(encoder_output)
        output = self.build_visualization(attn_output)

        return output

    def test_worker(self):
        '''
        Testing.
        '''
        logits, _ = self.build_pipe()
        logits = torch.softmax(logits, dim=2)

        ratePred = logits.topk(1, dim=2)[1].squeeze(2).data.cpu().numpy()
        ratePred += 1
        ratePred = ratePred.tolist()

        rateTrue = self.batch_data['rating'].data.cpu().numpy()
        rateTrue += 1
        rateTrue = rateTrue.tolist()

        return ratePred, rateTrue

    def test_penultimate_worker(self):
        '''
        Testing.
        '''
        logits, _ = self.build_pipe()

        return logits

    def build_keywords(self, input_):
        '''
        Keywords
        '''
        raise NotImplementedError

    def keywords_worker(self):
        '''
        Keywords worker
        '''
        encoder_output = self.build_encoder()
        attn_output = self.build_attention(encoder_output)
        keywords_output = self.build_keywords(attn_output)

        return keywords_output

    def run_evaluation(self):
        '''
        For evaluation.
        '''
        self.pred_data = np.array(self.pred_data)
        self.true_data = np.array(self.true_data)

        label_pred = []
        label_true = []
        for k in range(self.args.n_tasks):
            predlb = [rt for idx, rt in enumerate(
                self.pred_data[:, k].tolist()) if self.true_data[idx, k] != 0]
            truelb = [rt for idx, rt in enumerate(
                self.true_data[:, k].tolist()) if self.true_data[idx, k] != 0]
            label_pred += predlb
            label_true += truelb

        accu = accuracy_score(label_true, label_pred)
        mse = mean_squared_error(label_true, label_pred)
        accu = np.round(accu, 4)
        mse = np.round(mse, 4)

        print('Accuracy={}, MSE={}'.format(accu, mse))
        return accu
