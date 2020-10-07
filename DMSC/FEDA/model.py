'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from DMSC.model_base_pre_emb import modelDMSCBasePreEmb
from LeafNATS.modules.encoder.encoder_rnn import EncoderRNN
from LeafNATS.modules.utils.CompressionFM import CompressionFM


class modelDMSC(modelDMSCBasePreEmb):

    def __init__(self, args):
        super().__init__(args=args)

    def build_models(self):
        '''
        Build all models.
        '''
        self.base_models['embedding'] = torch.nn.Embedding(
            self.batch_data['vocab_size'], self.args.emb_dim
        ).to(self.args.device)

        self.train_models['gate'] = torch.nn.Linear(
            self.args.emb_dim, self.args.emb_dim
        ).to(self.args.device)
        self.train_models['value'] = torch.nn.Linear(
            self.args.emb_dim, self.args.emb_dim
        ).to(self.args.device)

        self.train_models['encoder'] = EncoderRNN(
            emb_dim=self.args.emb_dim,
            hidden_size=self.args.rnn_hidden_dim,
            nLayers=self.args.rnn_nLayers,
            rnn_network=self.args.rnn_network,
            device=self.args.device
        ).to(self.args.device)

        self.train_models['max_pool'] = torch.nn.MaxPool1d(
            self.args.rnn_hidden_dim, stride=1).to(self.args.device)
        self.train_models['avg_pool'] = torch.nn.AvgPool1d(
            self.args.rnn_hidden_dim, stride=1).to(self.args.device)

        self.train_models['fmf'] = CompressionFM(
            self.args.rnn_hidden_dim, self.args.rnn_hidden_dim*2).to(self.args.device)
        self.train_models['fmb'] = CompressionFM(
            self.args.rnn_hidden_dim, self.args.rnn_hidden_dim*2).to(self.args.device)

        self.train_models['attn_forward'] = torch.nn.ModuleList(
            [torch.nn.Linear(self.args.rnn_hidden_dim*2+6, self.args.rnn_hidden_dim*2)
             for k in range(self.args.n_tasks)]).to(self.args.device)
        self.train_models['attn_wrap'] = torch.nn.ModuleList(
            [torch.nn.Linear(self.args.rnn_hidden_dim*2, 1)
             for k in range(self.args.n_tasks)]).to(self.args.device)

        self.train_models['loop_forward1'] = torch.nn.ModuleList(
            [torch.nn.Linear(self.args.rnn_hidden_dim*2+6, self.args.rnn_hidden_dim*2+6)
             for k in range(self.args.n_tasks)]).to(self.args.device)

        self.train_models['ff'] = torch.nn.ModuleList(
            [torch.nn.Linear(self.args.rnn_hidden_dim*2+6, self.args.rnn_hidden_dim*2)
             for k in range(self.args.n_tasks)]).to(self.args.device)
        self.train_models['classifier'] = torch.nn.ModuleList(
            [torch.nn.Linear(self.args.rnn_hidden_dim*2, self.args.n_class)
             for k in range(self.args.n_tasks)]).to(self.args.device)

        self.train_models['drop'] = torch.nn.Dropout(self.args.drop_rate)

        self.loss_criterion = torch.nn.CrossEntropyLoss(
            ignore_index=-1).to(self.args.device)

    def build_pipe(self):
        '''
        Shared pipe
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
        for k in range(self.args.n_tasks):

            attn0 = torch.tanh(
                self.train_models['attn_forward'][k](review_enc))
            attn0 = self.train_models['attn_wrap'][k](attn0).squeeze(2)
            attn0 = attn0.masked_fill(
                self.batch_data['weight_mask'] == 0, -1e9)
            attn0 = torch.softmax(attn0, 1)
            cv_hidden0 = torch.bmm(attn0.unsqueeze(1), review_enc).squeeze(1)

            attn1 = torch.tanh(
                self.train_models['loop_forward1'][k](review_enc))
            attn1 = torch.bmm(attn1, cv_hidden0.unsqueeze(2)).squeeze(2)
            attn1 = attn1.masked_fill(
                self.batch_data['weight_mask'] == 0, -1e9)
            attn1 = torch.softmax(attn1, 1)
            cv_hidden1 = torch.bmm(attn1.unsqueeze(1), review_enc).squeeze(1)

            cv_hidden = cv_hidden0 + cv_hidden1
            fc = torch.relu(self.train_models['drop'](
                self.train_models['ff'][k](cv_hidden)))
            logits.append(self.train_models['drop'](
                self.train_models['classifier'][k](fc)))

        logits = torch.cat(logits, 0)
        logits = logits.view(self.args.n_tasks, batch_size, self.args.n_class)
        logits = logits.transpose(0, 1)

        return logits
