'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from FGDMSC.model_base_fine import modelFineBase
from LeafNATS.modules.attention.attention_self import AttentionSelf
from LeafNATS.modules.encoder.encoder_rnn import EncoderRNN


class modelFine(modelFineBase):

    def __init__(self, args):
        super().__init__(args=args)

    def build_models(self):
        '''
        Build all models.
        '''
        self.base_models['embedding'] = torch.nn.Embedding(
            self.batch_data['vocab_size'], self.args.emb_dim
        ).to(self.args.device)

        self.train_models['encoder'] = EncoderRNN(
            emb_dim=self.args.emb_dim,
            hidden_size=self.args.rnn_hidden_dim,
            nLayers=self.args.rnn_nLayers,
            rnn_network=self.args.rnn_network,
            device=self.args.device
        ).to(self.args.device)

        self.train_models['attn_aspect'] = AttentionSelf(
            input_size=self.args.rnn_hidden_dim*2,
            hidden_size=self.args.rnn_hidden_dim*2,
            device=self.args.device
        ).to(self.args.device)

        self.train_models['attn_sent'] = AttentionSelf(
            input_size=self.args.rnn_hidden_dim*2,
            hidden_size=self.args.rnn_hidden_dim*2,
            device=self.args.device
        ).to(self.args.device)

        self.train_models['ff_aspect'] = torch.nn.Linear(
            self.args.rnn_hidden_dim*2, self.args.rnn_hidden_dim*2
        ).to(self.args.device)
        self.train_models['classifier_aspect'] = torch.nn.Linear(
            self.args.rnn_hidden_dim*2, self.args.n_tasks
        ).to(self.args.device)

        self.train_models['ff_sent'] = torch.nn.Linear(
            self.args.rnn_hidden_dim*2, self.args.rnn_hidden_dim*2
        ).to(self.args.device)
        self.train_models['classifier_sent'] = torch.nn.Linear(
            self.args.rnn_hidden_dim*2, self.args.n_class
        ).to(self.args.device)

        self.train_models['drop'] = torch.nn.Dropout(self.args.drop_rate)

    def build_encoder(self):
        '''
        Encoder
        '''
        input_emb = self.base_models['embedding'](self.batch_data['input_ids'])
        input_enc, _ = self.train_models['encoder'](input_emb)

        return input_enc

    def build_attention(self, input_):
        '''
        Attention
        '''
        attn_aspect, ctx_aspect = self.train_models['attn_aspect'](
            input_, self.batch_data['pad_mask'])
        attn_sent, ctx_sent = self.train_models['attn_sent'](
            input_, self.batch_data['pad_mask'])

        output = {
            'attn_aspect': attn_aspect, 'ctx_aspect': ctx_aspect,
            'attn_sent': attn_sent, 'ctx_sent': ctx_sent}

        return output

    def build_classifier(self, input_):
        '''
        Classifier
        '''
        ctx_aspect = input_['ctx_aspect']
        ctx_sent = input_['ctx_sent']

        fc = torch.relu(self.train_models['drop'](
            self.train_models['ff_aspect'](ctx_aspect)))
        logits_aspect = self.train_models['drop'](
            self.train_models['classifier_aspect'](fc))
        fc = torch.relu(self.train_models['drop'](
            self.train_models['ff_sent'](ctx_sent)))
        logits_sent = self.train_models['drop'](
            self.train_models['classifier_sent'](fc))

        return [logits_aspect, logits_sent]

    def build_keywords(self, input_):
        '''
        Keywords
        '''
        input_ids = self.batch_data['input_ids'].data.cpu().numpy().tolist()
        input_text = []
        for k in range(len(input_ids)):
            out = []
            for j in range(len(input_ids[k])):
                if not input_ids[k][j] == 1:
                    out.append(self.batch_data['id2vocab'][input_ids[k][j]])
            input_text.append(out)

        attn_ = input_['attn']
        cand_words = attn_.topk(self.args.n_keywords)[
            1].data.cpu().numpy().tolist()
        cand_weights = attn_.topk(self.args.n_keywords)[
            0].data.cpu().numpy().tolist()
        cand_weights = np.around(cand_weights, 4).tolist()

        for k in range(len(cand_words)):
            for j in range(len(cand_words[k])):
                cand_words[k][j] = input_text[k][cand_words[k][j]]

        output = []
        for k in range(len(cand_words)):
            output.append({'toks': cand_words[k], 'weights': cand_weights[k]})

        return output

    def build_visualization(self, input_):
        '''
        visualization
        '''
        input_weights = input_['attn'].data.cpu().numpy().tolist()
        input_weights = np.around(input_weights, 4).tolist()

        input_ids = self.batch_data['input_ids'].data.cpu().numpy().tolist()
        output_text = []
        output_weights = []
        for k in range(len(input_ids)):
            out_text = []
            out_weight = []
            for j in range(len(input_ids[k])):
                if not input_ids[k][j] == 1:
                    out_text.append(
                        self.batch_data['id2vocab'][input_ids[k][j]])
                    out_weight.append(input_weights[k][j])
            output_text.append(out_text)
            output_weights.append(out_weight)

        output = []
        for k in range(len(output_text)):
            output.append(
                {'toks': output_text[k], 'weights': output_weights[k]})

        return output
