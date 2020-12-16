'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from DMSCFEDA.model_base import modelDMSCBase
from LeafNATS.modules.encoder.encoder_rnn import EncoderRNN
from LeafNATS.modules.utils.CompressionFM import CompressionFM


class modelCoarse(modelDMSCBase):

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
            [torch.nn.Linear(self.args.rnn_hidden_dim*2+7, self.args.rnn_hidden_dim*2)
             for k in range(self.args.n_tasks)]).to(self.args.device)
        self.train_models['classifier'] = torch.nn.ModuleList(
            [torch.nn.Linear(self.args.rnn_hidden_dim*2+1, self.args.n_class)
             for k in range(self.args.n_tasks)]).to(self.args.device)

        self.train_models['drop'] = torch.nn.Dropout(self.args.drop_rate)

    def build_encoder(self):
        '''
        Encoder
        '''
        review_emb = self.base_models['embedding'](self.batch_data['review'])
        batch_size = review_emb.size(0)
        seq_len = review_emb.size(1)
        emb_gate = torch.sigmoid(self.train_models['gate'](review_emb))
        emb_gate = self.train_models['drop'](emb_gate)
        emb_valu = torch.relu(self.train_models['value'](review_emb))
        emb_valu = self.train_models['drop'](emb_valu)
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
        review_enc = self.train_models['drop'](review_enc)

        return review_enc

    def build_attention(self, input_):
        '''
        Attention
        '''
        attn_arr = []
        ctx_arr = []
        for k in range(self.args.n_tasks):
            attn0 = torch.tanh(
                self.train_models['attn_forward'][k](input_))
            attn0 = self.train_models['attn_wrap'][k](attn0).squeeze(2)
            attn0 = attn0.masked_fill(
                self.batch_data['weight_mask'] == 0, -1e9)
            attn0 = torch.softmax(attn0, 1)
            cv_hidden0 = torch.bmm(attn0.unsqueeze(1), input_).squeeze(1)

            attn1 = torch.tanh(
                self.train_models['loop_forward1'][k](input_))
            attn1 = torch.bmm(attn1, cv_hidden0.unsqueeze(2)).squeeze(2)
            attn1 = attn1.masked_fill(
                self.batch_data['weight_mask'] == 0, -1e9)
            attn1 = torch.softmax(attn1, 1)

            attnA = attn0 + attn1
            if self.args.task == 'test_uncertainty' \
                    and self.args.drop_option == 'attention':
                attnA = self.train_models['drop'](
                    attnA)/torch.sum(attnA, dim=1).unsqueeze(1)*2.0
            cv_hiddenA = torch.bmm(attnA.unsqueeze(1), input_).squeeze(1)

            attn_arr.append(0.5*attnA)
            ctx_arr.append(cv_hiddenA)

        output = {'attn': attn_arr, 'ctx': ctx_arr}

        return output

    def build_classifier(self, input_):
        '''
        Classifier
        '''
        ctx_arr = input_['ctx']
        batch_size = ctx_arr[0].size(0)

        logits = []
        review_vec = []
        for k in range(self.args.n_tasks):
            if self.args.task == 'test_uncertainty':
                fc = torch.cat([ctx_arr[k], self.batch_data['rateOV']], 1)
                if self.args.drop_option == 'vector':
                    fc = torch.relu(self.train_models['drop'](
                        self.train_models['ff'][k](fc)))
                else:
                    fc = torch.relu(self.train_models['ff'][k](fc))
                fc = torch.cat([fc, self.batch_data['rateOV']], 1)
                fc = self.train_models['classifier'][k](fc)
            else:
                fc = torch.cat([ctx_arr[k], self.batch_data['rateOV']], 1)
                fc = torch.relu(self.train_models['drop'](
                    self.train_models['ff'][k](fc)))
                review_vec.append(fc)
                fc = torch.cat([fc, self.batch_data['rateOV']], 1)
                fc = self.train_models['drop'](
                    self.train_models['classifier'][k](fc))
            logits.append(fc)

        logits = torch.cat(logits, 0)
        logits = logits.view(
            self.args.n_tasks, batch_size, self.args.n_class)
        logits = logits.transpose(0, 1)

        return logits, review_vec

    def metric_learning(self, input_):
        '''
        metric learning
        '''
        rating = self.batch_data['rating']
        loss_intra = []
        loss_inter = []
        for k in range(self.args.n_tasks):
            batch_size, vec_size = input_[k].size()
            out_intra = []
            out_inter = []
            for j in range(batch_size):
                for i in range(j+1):
                    if rating[j, k] == -1 or rating[i, k] == -1:
                        continue
                    dist_ij = torch.dist(input_[k][j], input_[k][i], 2)
                    norm_ij = (dist_ij * dist_ij) / vec_size
                    if rating[j, k] == rating[i, k]:
                        out_intra.append(norm_ij)
                    else:
                        out_inter.append(norm_ij)
            if len(out_intra) > 0:
                loss_intra.append(sum(out_intra)/len(out_intra))
            else:
                loss_intra.append(sum(out_intra))
            if len(out_inter) > 0:
                loss_inter.append(
                    torch.relu(0.1 - sum(out_inter)/len(out_inter)))
            else:
                loss_inter.append(0.1 - sum(out_inter))

        return (sum(loss_intra)+0.1*sum(loss_inter))/self.args.n_tasks

    def build_keywords(self, input_):
        '''
        Keywords
        '''
        input_ids = self.batch_data['review'].data.cpu().numpy().tolist()
        input_text = []
        for k in range(len(input_ids)):
            out = []
            for j in range(len(input_ids[k])):
                if not input_ids[k][j] == 1:
                    out.append(self.batch_data['id2vocab'][input_ids[k][j]])
            input_text.append(out)

        arr_words = []
        arr_weights = []
        batch_size = input_['attn'][0].size(0)
        for idx in range(self.args.n_tasks):

            attn_ = input_['attn'][idx]
            cand_words = attn_.topk(
                self.args.n_keywords)[1].data.cpu().numpy().tolist()
            cand_weights = attn_.topk(
                self.args.n_keywords)[0].data.cpu().numpy().tolist()
            cand_weights = np.around(cand_weights, 4).tolist()

            for k in range(batch_size):
                for j in range(len(cand_words[k])):
                    try:
                        cand_words[k][j] = input_text[k][cand_words[k][j]]
                    except:
                        continue

            arr_words.append(cand_words)
            arr_weights.append(cand_weights)

        output = []
        for k in range(batch_size):
            out_words = []
            out_weights = []
            for j in range(self.args.n_tasks):
                out_words.append(arr_words[j][k])
                out_weights.append(arr_weights[j][k])
            output.append({'toks': out_words, 'weights': out_weights})

        return output

    def build_visualization(self, input_):

        attn = input_['attn']
        batch_size = attn[0].size(0)

        output = []
        for k in range(batch_size):
            txt = []
            wt = []
            tt = self.batch_data['review_txt'][k]
            for j in range(self.args.n_tasks):
                aa = attn[j][k].data.cpu().numpy().tolist()[:len(tt)]
                wt.append(aa)
                txt.append(' '.join(self.batch_data['review_txt'][k]))
            output.append({'text': txt, 'weights': wt})

        return output
