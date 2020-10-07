'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import glob
import json
import os
import pickle
import re
import shutil
import time
from pprint import pprint

import numpy as np
import torch
from torch.autograd import Variable

from LeafNATS.data.utils import create_batch_memory, load_vocab_pretrain
from LeafNATS.engines.end2end_small import End2EndBase
from LeafNATS.utils.utils import show_progress

from .visualization_module import createHTML


class End2EndBaseDMSCFine(End2EndBase):
    '''
    End2End training classification.
    Not suitable for language generation task.
    Light weight. Data should be relevatively small.
    '''

    def __init__(self, args=None):
        '''
        Initialize
        '''
        super().__init__(args=args)

    def build_scheduler(self, optimizer):
        '''
        Schedule Learning Rate
        '''
        if self.args.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer, step_size=self.args.step_size,
                gamma=self.args.step_decay)

        return scheduler

    def build_optimizer(self, params):
        '''
        init model optimizer
        '''
        self.loss_criterion = torch.nn.CrossEntropyLoss(
            ignore_index=-1).to(self.args.device)

        optimizer = torch.optim.Adam(
            params, lr=self.args.learning_rate)

        return optimizer

    def build_vocabulary(self):
        '''
        vocabulary
        '''
        vocab2id, id2vocab, pretrain_vec = load_vocab_pretrain(
            os.path.join(self.args.data_dir, self.args.file_pretrain_vocab),
            os.path.join(self.args.data_dir, self.args.file_pretrain_vec))
        vocab_size = len(vocab2id)
        self.batch_data['vocab2id'] = vocab2id
        self.batch_data['id2vocab'] = id2vocab
        self.batch_data['pretrain_emb'] = pretrain_vec
        self.batch_data['vocab_size'] = vocab_size
        print('The vocabulary size: {}'.format(vocab_size))

    def init_base_model_params(self):
        '''
        Initialize Base Model Parameters.
        '''
        emb_para = torch.FloatTensor(
            self.batch_data['pretrain_emb']).to(self.args.device)
        self.base_models['embedding'].weight = torch.nn.Parameter(emb_para)

        for model_name in self.base_models:
            if model_name == 'embedding':
                continue
            fl_ = os.path.join(self.args.base_model_dir, model_name+'.model')
            self.base_models[model_name].load_state_dict(
                torch.load(fl_, map_location=lambda storage, loc: storage))

    def test_worker(self):
        '''
        Used in decoding.
        Users can define their own decoding process.
        You do not have to worry about path and prepare input.
        '''
        raise NotImplementedError

    def keywords_worker(self):
        '''
        Keywords worker
        '''
        raise NotImplementedError

    def visualization_worker(self):
        '''
        Visualization worker
        '''
        raise NotImplementedError

    def test(self):
        '''
        Testing
        '''
        self.build_vocabulary()
        self.build_models()
        print(self.base_models)
        print(self.train_models)
        if len(self.base_models) > 0:
            self.init_base_model_params()
        if len(self.train_models) > 0:
            self.init_train_model_params()

        self.test_data = create_batch_memory(
            path_=self.args.data_dir,
            file_=self.args.file_test,
            is_shuffle=False,
            batch_size=self.args.batch_size,
            is_lower=self.args.is_lower)

        output_dir = '../nats_results/' + self.args.test_output_dir
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for model_name in self.base_models:
            self.base_models[model_name].eval()
        for model_name in self.train_models:
            self.train_models[model_name].eval()

        with torch.no_grad():
            print('Begin Testing: {}'.format(self.args.file_test))
            test_batch = len(self.test_data)
            print(
                'The number of batches (testing): {}'.format(test_batch))
            self.pred_data = []
            self.true_data = []
            if self.args.debug:
                test_batch = 3
            for test_id in range(test_batch):
                self.build_batch(self.test_data[test_id])
                ratePred, rateTrue = self.test_worker()

                self.pred_data += ratePred
                self.true_data += rateTrue

                show_progress(test_id+1, test_batch)
            print()
            # save testing data.
            try:
                self.pred_data = np.array(
                    self.pred_data).astype(int)
                np.savetxt(
                    os.path.join(
                        output_dir, '{}_pred_{}.txt'.format(
                            self.args.file_test, self.args.best_epoch)),
                    self.pred_data, fmt='%d')
                self.true_data = np.array(self.true_data).astype(int)
                np.savetxt(
                    os.path.join(
                        output_dir, '{}_true_{}.txt'.format(
                            self.args.file_test, self.args.best_epoch)),
                    self.true_data, fmt='%d')
            except:
                fout = open(os.path.join(
                    output_dir,
                    '{}_pred_{}.pickled'.format(
                        self.args.file_best, self.args.best_epoch)), 'wb')
                pickle.dump(self.pred_data, fout)
                fout.close()
                fout = open(os.path.join(
                    output_dir,
                    '{}_true_{}.pickled'.format(
                        self.args.file_test, self.args.best_epoch)), 'wb')
                pickle.dump(self.true_data, fout)
                fout.close()

    def test_penultimate(self):
        '''
        Testing
        '''
        self.build_vocabulary()
        self.build_models()
        print(self.base_models)
        print(self.train_models)
        if len(self.base_models) > 0:
            self.init_base_model_params()
        if len(self.train_models) > 0:
            self.init_train_model_params()

        self.test_data = create_batch_memory(
            path_=self.args.data_dir,
            file_=self.args.file_test,
            is_shuffle=False,
            batch_size=self.args.batch_size,
            is_lower=self.args.is_lower)

        output_dir = '../nats_results/' + \
            self.args.test_output_dir + '_penultimate'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for model_name in self.base_models:
            self.base_models[model_name].eval()
        for model_name in self.train_models:
            self.train_models[model_name].eval()

        with torch.no_grad():
            print('Begin Testing: {}'.format(self.args.file_test))
            test_batch = len(self.test_data)
            print(
                'The number of batches (testing): {}'.format(test_batch))
            self.pred_data = []
            if self.args.debug:
                test_batch = 3
            for test_id in range(test_batch):
                self.build_batch(self.test_data[test_id])
                logits = self.test_penultimate_worker()
                self.pred_data += logits.data.cpu().numpy().tolist()

                show_progress(test_id+1, test_batch)
            print()
            # save testing data.
            outfile = os.path.join(output_dir, '{}_pred_{}.pickled'.format(
                self.args.file_test, self.args.best_epoch))
            fout = open(outfile, 'wb')
            pickle.dump(self.pred_data, fout)
            fout.close()

    def test_uncertainty(self):
        '''
        Testing Uncertainty
        '''
        self.build_vocabulary()
        self.build_models()
        print(self.base_models)
        print(self.train_models)
        if len(self.base_models) > 0:
            self.init_base_model_params()
        if len(self.train_models) > 0:
            self.init_train_model_params()

        self.test_data = create_batch_memory(
            path_=self.args.data_dir,
            file_=self.args.file_test,
            is_shuffle=False,
            batch_size=self.args.batch_size,
            is_lower=self.args.is_lower)

        output_dir = '../nats_results/{}_uncertainty_{}_{}'.format(
            self.args.test_output_dir,
            self.args.drop_option,
            self.args.drop_rate)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        with torch.no_grad():
            for k_unc in range(self.args.uncertainty_total_samples):

                print('Begin Testing: {}, {}'.format(
                    self.args.file_test, k_unc))
                test_batch = len(self.test_data)
                print('The number of batches (testing): {}'.format(
                    test_batch))
                self.pred_data = []
                self.true_data = []
                self.vector_data = []
                if self.args.debug:
                    test_batch = 3
                for test_id in range(test_batch):
                    self.build_batch(self.test_data[test_id])
                    logits = self.test_penultimate_worker()
                    ratePred, rateTrue = self.test_worker()

                    self.vector_data += logits.data.cpu().numpy().tolist()
                    self.pred_data += ratePred
                    self.true_data += rateTrue

                    show_progress(test_id+1, test_batch)
                print()
                # save testing data.
                outfile = os.path.join(output_dir, '{}_vector_{}_{}.pickled'.format(
                    self.args.file_test, self.args.best_epoch, k_unc))
                fout = open(outfile, 'wb')
                pickle.dump(self.vector_data, fout)
                fout.close()
                try:
                    self.pred_data = np.array(
                        self.pred_data).astype(int)
                    np.savetxt(
                        os.path.join(
                            output_dir, '{}_pred_{}_unc_{}.txt'.format(
                                self.args.file_test, self.args.best_epoch, k_unc)),
                        self.pred_data, fmt='%d')
                    self.true_data = np.array(self.true_data).astype(int)
                    np.savetxt(
                        os.path.join(
                            output_dir, '{}_true_{}_unc_{}.txt'.format(
                                self.args.file_test, self.args.best_epoch, k_unc)),
                        self.true_data, fmt='%d')
                except:
                    fout = open(os.path.join(
                        output_dir,
                        '{}_pred_{}.pickled'.format(
                            self.args.file_best, self.args.best_epoch)), 'wb')
                    pickle.dump(self.pred_data, fout)
                    fout.close()
                    fout = open(os.path.join(
                        output_dir,
                        '{}_true_{}.pickled'.format(
                            self.args.file_test, self.args.best_epoch)), 'wb')
                    pickle.dump(self.true_data, fout)
                    fout.close()

    def keywords(self):
        '''
        Keywords SelfAttention
        '''
        self.build_vocabulary()
        self.build_models()
        print(self.base_models)
        print(self.train_models)
        if len(self.base_models) > 0:
            self.init_base_model_params()
        if len(self.train_models) > 0:
            self.init_train_model_params()

        self.test_data = create_batch_memory(
            path_=self.args.data_dir,
            file_=self.args.file_test,
            is_shuffle=False,
            batch_size=self.args.batch_size,
            is_lower=self.args.is_lower)

        output_dir = '../nats_results/' + self.args.keywords_output_dir
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for model_name in self.base_models:
            self.base_models[model_name].eval()
        for model_name in self.train_models:
            self.train_models[model_name].eval()

        with torch.no_grad():
            print('Begin Testing: {}'.format(self.args.file_test))
            test_batch = len(self.test_data)
            print(
                'The number of batches (testing): {}'.format(test_batch))
            pred_data = []
            true_data = []
            keywords_data = []
            if self.args.debug:
                test_batch = 3
            for test_id in range(test_batch):
                self.build_batch(self.test_data[test_id])
                ratePred, rateTrue = self.test_worker()
                output = self.keywords_worker()
                keywords_data += output

                pred_data += ratePred
                true_data += rateTrue

                show_progress(test_id+1, test_batch)
            print()

            for k in range(len(keywords_data)):
                keywords_data[k]['pred_label'] = pred_data[k]
                keywords_data[k]['gold_label'] = true_data[k]

            fout = open(os.path.join(
                output_dir,
                '{}_{}.pickled'.format(
                    self.args.file_test, self.args.best_epoch)), 'wb')
            pickle.dump(keywords_data, fout)
            fout.close()

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

        self.test_data = create_batch_memory(
            path_=self.args.data_dir,
            file_=self.args.file_test,
            is_shuffle=False,
            batch_size=self.args.batch_size,
            is_lower=self.args.is_lower
        )

        for model_name in self.base_models:
            self.base_models[model_name].eval()
        for model_name in self.train_models:
            self.train_models[model_name].eval()

        output_dir = '../nats_results/visualization_{}'.format(
            self.args.file_test)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        aspect_label = self.args.aspect_label.split(',')
        vis_label = [int(wd)-1 for wd in self.args.visualization_aspect.split(',')]
        data_aspect = [aspect_label[idx] for idx in vis_label]
        print('You will visualize Aspects: {}'.format(', '.join(data_aspect)))

        with torch.no_grad():
            print('Begin Testing: {}'.format(self.args.file_test))
            test_batch = len(self.test_data)
            print(
                'The number of batches (testing): {}'.format(test_batch))
            pred_data = []
            true_data = []
            keywords_data = []
            if self.args.debug:
                test_batch = 3
            for test_id in range(test_batch):
                self.build_batch(self.test_data[test_id])
                ratePred, rateTrue = self.test_worker()
                output = self.visualization_worker()
                keywords_data += output

                pred_data += ratePred
                true_data += rateTrue

                show_progress(test_id+1, test_batch)
            print()
            for k in range(len(keywords_data)):
                keywords_data[k]['pred_label'] = [pred_data[k][idx]
                                                  for idx in vis_label]
                keywords_data[k]['gold_label'] = [true_data[k][idx]
                                                  for idx in vis_label]
                len_txt = len(keywords_data[k]['text'][0].split())
                diff = []
                for j in range(len(pred_data[k])):
                    if pred_data[k][j] == true_data[k][j] and true_data[k][j] > 0:
                        diff.append(0)
                    else:
                        diff.append(1)
                diff = np.sum(diff)
                ftxt = '_'.join(
                    map(str, [k, len_txt, diff] + true_data[k] + pred_data[k]))
                file_output = os.path.join(output_dir, '{}.html'.format(ftxt))
                keywords_data[k]['text'] = [keywords_data[k]['text'][idx]
                                            for idx in vis_label]
                keywords_data[k]['weights'] = [keywords_data[k]['weights'][idx]
                                            for idx in vis_label]
                createHTML(data_aspect, keywords_data[k], file_output)