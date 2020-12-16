'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import os
import time
from pprint import pprint

import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_squared_error
from torch.autograd import Variable

from LeafNATS.data.utils import construct_vocab

from .data.process_minibatch_fine import (process_minibatch,
                                          process_minibatch_test)
from .end2end_engine import End2EndBaseFGDMSC


class modelFineBase(End2EndBaseFGDMSC):
    '''
    Fine ABSA.
    '''

    def __init__(self, args):
        super().__init__(args=args)

    def build_batch(self, batch_):
        '''
        get batch data
        '''
        if self.args.task == 'train':
            review, pad_mask, aspect_label, sent_label = process_minibatch(
                batch_, self.batch_data['vocab2id'],
                self.args.review_max_lens)

            self.batch_data['input_ids'] = review.to(self.args.device)
            self.batch_data['pad_mask'] = pad_mask.to(self.args.device)
            self.batch_data['aspect_label'] = aspect_label.to(self.args.device)
            self.batch_data['sent_label'] = sent_label.to(self.args.device)
        else:
            review, pad_mask, label, text, ids = process_minibatch_test(
                batch_, self.batch_data['vocab2id'],
                self.args.review_max_lens)
            self.batch_data['label'] = label
            self.batch_data['review_text'] = text
            self.batch_data['review_ids'] = ids

            self.batch_data['tmp'] = {}
            self.batch_data['tmp']['input_ids'] = [
                itm.to(self.args.device) for itm in review]
            self.batch_data['tmp']['pad_mask'] = [
                itm.to(self.args.device) for itm in pad_mask]

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
        logits = self.build_classifier(attn_output)

        return logits

    def build_pipelines(self):
        '''
        Data flow from input to output.
        '''
        logits = self.build_pipe()
        logits_aspect = logits[0]
        logits_sent = logits[1]

        loss_aspect = self.loss_criterion(
            logits_aspect, self.batch_data['aspect_label'].view(-1))
        loss_sent = self.loss_criterion(
            logits_sent, self.batch_data['sent_label'].view(-1))

        return 0.5*(loss_aspect+loss_sent)

    def test_worker(self):
        '''
        Testing worker
        '''
        if self.args.task == 'train':
            logits = self.build_pipe()
            logits_aspect = logits[0]
            logits_sent = logits[1]
            prob_aspect = torch.softmax(logits_aspect, dim=1)
            prob_sent = torch.softmax(logits_sent, dim=1)

            rate_aspect = prob_aspect.topk(
                1, dim=1)[1].squeeze(1).data.cpu().numpy()
            rate_aspect = rate_aspect.tolist()
            rate_sent = prob_sent.topk(
                1, dim=1)[1].squeeze(1).data.cpu().numpy()
            rate_sent = rate_sent.tolist()
            ratePred = [[rate_aspect[k], rate_sent[k]]
                        for k in range(len(rate_aspect))]

            rate_aspect = self.batch_data['aspect_label'].data.cpu().numpy()
            rate_aspect = rate_aspect.tolist()
            rate_sent = self.batch_data['sent_label'].data.cpu().numpy()
            rate_sent = rate_sent.tolist()
            rateTrue = [[rate_aspect[k], rate_sent[k]]
                        for k in range(len(rate_aspect))]

            return ratePred, rateTrue
        else:
            n_bb = len(self.batch_data['tmp']['input_ids'])
            ratePred = []
            for k in range(n_bb):
                self.batch_data['input_ids'] = \
                    self.batch_data['tmp']['input_ids'][k]
                self.batch_data['pad_mask'] = \
                    self.batch_data['tmp']['pad_mask'][k]

                logits = self.build_pipe()
                logits_aspect = logits[0]
                logits_sent = logits[1]
                prob_aspect = torch.softmax(logits_aspect, dim=1)
                prob_sent = torch.softmax(logits_sent, dim=1)

                rate_aspect = prob_aspect.topk(
                    1, dim=1)[1].squeeze(1).data.cpu().numpy()
                rate_aspect = rate_aspect.tolist()
                rate_sent = prob_sent.topk(
                    1, dim=1)[1].squeeze(1).data.cpu().numpy()
                rate_sent = rate_sent.tolist()

                rate_out = {}
                text_out = []
                for j in range(len(rate_aspect)):
                    try:
                        rate_out[rate_aspect[j]].append(rate_sent[j])
                    except:
                        rate_out[rate_aspect[j]] = [rate_sent[j]]
                    # output text
                    text_out.append({
                        'id': '{} {}'.format(self.batch_data['review_ids'][k], j),
                        'text': ' '.join(self.batch_data['review_text'][k][j].split()),
                        'aspect': rate_aspect[j],
                        'sentiment': 'positive' if rate_sent[j] == 1 else 'negative'
                    })
                self.review_details.append(text_out)

                rate_out = {wd: np.mean(rate_out[wd]) for wd in rate_out}
                rp = []
                for j in range(self.args.n_tasks):
                    if j in rate_out:
                        if rate_out[j] > 0.5:
                            rp.append(2)
                        else:
                            rp.append(1)
                    else:
                        rp.append(0)
                ratePred.append(rp)

            rateTrue = self.batch_data['label'].data.cpu().numpy()
            rateTrue += 1
            rateTrue = rateTrue.tolist()

            return ratePred, rateTrue

    def build_keywords(self, input_):
        '''
        Keywords
        '''
        raise NotImplementedError

    def build_visualization(self, input_):
        '''
        visualization
        '''
        raise NotImplementedError

    def keywords_worker(self):
        '''
        Keywords worker
        '''
        output_enc = self.build_encoder()
        output_attn = self.build_attention(output_enc)
        output_keywords = self.build_keywords_attnself(output_attn)

        return output_keywords

    def visualization_worker(self):
        '''
        visualization worker
        '''
        output_enc = self.build_encoder()
        output_attn = self.build_attention(output_enc)
        output = self.build_visualization_attnself(output_attn)

        return output

    def run_evaluation(self):
        '''
        For evaluation.
        '''
        self.pred_data = np.array(self.pred_data)
        self.true_data = np.array(self.true_data)

        label_pred = [nm for nm in self.pred_data[:, 0].tolist()]
        label_true = [nm for nm in self.true_data[:, 0].tolist()]
        accu = accuracy_score(label_true, label_pred)
        mse = mean_squared_error(label_true, label_pred)
        accu = np.round(accu, 4)
        mse = np.round(mse, 4)
        print('Aspect Prediction: Accuracy={}, MSE={}'.format(accu, mse))

        label_pred = [nm for nm in self.pred_data[:, 1].tolist()]
        label_true = [nm for nm in self.true_data[:, 1].tolist()]
        accu = accuracy_score(label_true, label_pred)
        mse = mean_squared_error(label_true, label_pred)
        accu = np.round(accu, 4)
        mse = np.round(mse, 4)
        print('Sentiment Prediction: Accuracy={}, MSE={}'.format(accu, mse))

        return accu
