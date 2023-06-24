from __future__ import absolute_import, division, print_function, unicode_literals
import json
import logging
import math
import os
import sys
from io import open
import torch
from torch import nn
from .encoder import TransformerInterEncoder, DecoderLayer, Decoder, Encoder, EncoderLayer
from .neural import positional_encodings_like
import torch.nn.functional as F
from transformers import AutoModel, AlbertModel, BertTokenizer, BertForPreTraining, BertModel, BartModel
from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
}

def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error("Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model

class BertForOrdering(nn.Module):

    def __init__(self, args):
        super(BertForOrdering, self).__init__()

        self.hidden_size = 768
        self.hidden_size_ba = 1024
        self.bart = BartModel.from_pretrained("facebook/bart-large")
        self.dropout = nn.Dropout(0.1)
        #self.encoder = TransformerInterEncoder(self.hidden_size, args.ff_size, args.heads, args.para_dropout,
                                               #args.inter_layers)
        selfatt_layer = EncoderLayer(self.hidden_size, args.heads, args.ff_size, args.para_dropout)
        self.encoder = Encoder(selfatt_layer, args.inter_layers)

        self.linear_hidden = nn.Linear(self.hidden_size_ba, self.hidden_size)
        self.key_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.query_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.tanh_linear = nn.Linear(self.hidden_size, 1)

        decodlayer = DecoderLayer(self.hidden_size, args.heads, self.hidden_size, args.para_dropout)
        self.decoder = Decoder(decodlayer, 1)
        self.critic = None

        self.pol = nn.Linear(self.hidden_size, self.hidden_size)


    def equip(self, critic):
        self.critic = critic

    def forward(self, input_ids, attention_mask=None, passage_length=None, ground_truth=None, s_index=None,
                cuda=None, head_mask=None):

        document_matrix, para_matrix, pos_emb, sents_mask, original_key = self.encode(
            input_ids, attention_mask, passage_length, ground_truth, s_index, cuda, head_mask)  #

        num_sen = passage_length
        order = ground_truth

        target = order
        tgt_len = num_sen
        batch, num = target.size()

        dec_outputs = self.decoder(pos_emb, para_matrix, sents_mask, sents_mask)
        query = self.query_linear(dec_outputs).unsqueeze(2)
        original_key = original_key.unsqueeze(1)
        e = torch.tanh(query + original_key)
        e = self.tanh_linear(e).squeeze(-1)
        pointed_mask = [e.new_zeros(e.size(0), 1, e.size(2)).bool()]
        for t in range(1, e.size(1)):
            tar = target[:, t - 1]
            pm = pointed_mask[-1].clone().detach()
            pm[torch.arange(e.size(0)), :, tar] = 1
            pointed_mask.append(pm)
        pointed_mask = torch.cat(pointed_mask, 1)
        pointed_mask_by_target = pointed_mask.new_zeros(pointed_mask.size(0), pointed_mask.size(2))
        target_mask = pointed_mask.new_zeros(pointed_mask.size(0), pointed_mask.size(1))

        for b in range(target_mask.size(0)):
            pointed_mask_by_target[b, :tgt_len[b]] = 1
            target_mask[b, :tgt_len[b]] = 1

        pointed_mask_by_target = pointed_mask_by_target.unsqueeze(1).expand_as(pointed_mask)

        e.masked_fill_(pointed_mask == 1, -1e9)
        e.masked_fill_(pointed_mask_by_target == 0, -1e9)

        logp = F.log_softmax(e, dim=-1)
        logp = logp.view(-1, logp.size(-1))
        loss = self.critic(logp, target.contiguous().view(-1))

        target_mask = target_mask.view(-1)
        loss.masked_fill_(target_mask == 0, 0)
        transform_loss = loss.reshape(batch, num)
        transform_loss_sample = torch.sum(transform_loss, -1) / (num_sen.float() + 1e-20 - 1)
        #transform_loss_sample = torch.sum(transform_loss, -1) / (num_sen.float())
        new_original_loss = transform_loss_sample.sum() / batch

        e1 = torch.tanh(query + original_key)
        e1 = self.tanh_linear(e1).squeeze(-1)
        e1.masked_fill_(pointed_mask_by_target == 0, -1e9)
        pointed_mask_by_target_col = torch.zeros_like(pointed_mask_by_target)
        for i in range(pointed_mask_by_target.size(0)):
            pointed_mask_by_target_col[i] = pointed_mask_by_target[i].t()
        e1.masked_fill_(pointed_mask_by_target_col == 0, -1e9)

        loss_col = 0
        logp2 = F.log_softmax(e1, dim=1)
        for i in range(batch):
            len1 = tgt_len[i]
            tar1 = target[i][:len1]
            pre_p1 = logp2[i, :len1, :len1].T
            m0 = tar1[0]
            col1 = pre_p1[m0].unsqueeze(0)
            for b in range(len1 - 1):
                m = tar1[b + 1]
                tgt_col = pre_p1[m].unsqueeze(0)
                col1 = torch.cat((col1, tgt_col), 0)
            collu = col1
            label = torch.linspace(0, len1 - 1, len1).long().cuda(int(cuda[-1]))
            loss2 = nn.NLLLoss()(collu, label) / (len1 + 1e-20 - 1)
            loss_col += loss2
        loss_col = loss_col / batch

        loss = new_original_loss + loss_col
        return loss

    def encode(self, input_ids, attention_mask=None, passage_length=None, ground_truth=None,
               s_index=None, cuda=None, head_mask=None):

        batch_size, max_sequence_len = input_ids.size()

        outputs = self.bart(input_ids,
                            attention_mask=attention_mask,
                            )

        last_hidden_states = outputs.last_hidden_state
        sents_vec = []
        for i in range(batch_size):
            index = s_index[i]
            s_hidden_states = last_hidden_states[i, index, :]
            sents_vec.append(s_hidden_states)
        sen_vec = torch.stack(sents_vec)
        sen_vec = self.linear_hidden(sen_vec)

        pos_emb_in = positional_encodings_like(sen_vec, t=None)

        pos_emb = self.pol(pos_emb_in)

        sents_mask = sen_vec.new_zeros(batch_size, sen_vec.size(1)).bool()

        for i in range(batch_size):
            sents_mask[i, :passage_length[i]] = 1
        sents_mask = sents_mask.unsqueeze(1)

        para_matrix = self.encoder(sen_vec, sents_mask)
        keyinput = sen_vec
        key = self.key_linear(keyinput)

        return sen_vec, para_matrix, pos_emb, sents_mask, key

def pointer_eval(args, model, input_ids, attention_mask=None, passage_length=None, ground_truth=None,
                       s_index=None, cuda=None, head_mask=None):
    sentences, para_matrix, pos_emb, sents_mask, original_keys = model.encode(
        input_ids, attention_mask, passage_length, ground_truth, s_index, cuda, head_mask)

    num_sen = passage_length
    batch, num = ground_truth.size()
    keys = original_keys[:, :num_sen, :]
    dec_output = model.decoder(pos_emb, para_matrix, sents_mask, sents_mask)
    query = model.query_linear(dec_output).unsqueeze(2)
    original_key = keys.unsqueeze(1)
    e = torch.tanh(query + original_key)
    e = model.tanh_linear(e).squeeze(-1)
    logp = F.softmax(e, dim=-1)
    log_p = logp.squeeze(0)
    mask = torch.zeros_like(log_p).bool()
    bestout = []
    for i in range(e.size(1)):
        best_p = torch.max(log_p[i].unsqueeze(0), 1)[1]
        m = best_p[0].item()
        bestout.append(m)
        mask[:, m] = 1
        log_p.masked_fill_(mask == 1, 0)
    best_output = bestout

    return best_output