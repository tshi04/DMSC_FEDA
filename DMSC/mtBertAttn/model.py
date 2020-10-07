'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from DMSC.model_base_bert import modelDMSCBaseBert
from LeafNATS.modules.encoder.encoder_rnn import EncoderRNN
from LeafNATS.modules.utils.CompressionFM import CompressionFM
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class modelDMSC(modelDMSCBaseBert):

    def __init__(self, args):
        super().__init__(args=args)

    def build_models(self):
        '''
        Build all models.
        '''
        self.pretrained_models['bert'] = BertModel.from_pretrained(
            'bert-base-uncased',
            output_hidden_states=True,
            output_attentions=True
        ).to(self.args.device)

        self.train_models['attn_forward'] = torch.nn.ModuleList(
            [torch.nn.Linear(768, 768)
             for k in range(self.args.n_tasks)]).to(self.args.device)
        self.train_models['attn_wrap'] = torch.nn.ModuleList(
            [torch.nn.Linear(768, 1)
             for k in range(self.args.n_tasks)]).to(self.args.device)

        self.train_models['ff'] = torch.nn.ModuleList(
            [torch.nn.Linear(768, 768)
             for k in range(self.args.n_tasks)]).to(self.args.device)
        self.train_models['classifier'] = torch.nn.ModuleList(
            [torch.nn.Linear(768, self.args.n_class)
             for k in range(self.args.n_tasks)]).to(self.args.device)

        self.train_models['drop'] = torch.nn.Dropout(self.args.drop_rate)

        self.loss_criterion = torch.nn.CrossEntropyLoss(
            ignore_index=-1).to(self.args.device)

    def build_pipe(self):
        '''
        Shared pipe
        '''
        batch_size = self.batch_data['input_ids'].size(0)
        with torch.no_grad():
            review_enc = self.pretrained_models['bert'](
                self.batch_data['input_ids'],
                self.batch_data['pad_mask'],
                self.batch_data['seg'])[0]

        logits = []
        for k in range(self.args.n_tasks):
            attn0 = torch.tanh(
                self.train_models['attn_forward'][k](review_enc))
            attn0 = self.train_models['attn_wrap'][k](attn0).squeeze(2)
            attn0 = attn0.masked_fill(
                self.batch_data['att_mask'] == 0, -1e9)
            attn0 = torch.softmax(attn0, 1)
            cv_hidden = torch.bmm(attn0.unsqueeze(1), review_enc).squeeze(1)

            fc = torch.relu(self.train_models['drop'](
                self.train_models['ff'][k](cv_hidden)))
            logits.append(self.train_models['drop'](
                self.train_models['classifier'][k](fc)))

        logits = torch.cat(logits, 0)
        logits = logits.view(self.args.n_tasks, batch_size, self.args.n_class)
        logits = logits.transpose(0, 1)

        return logits
