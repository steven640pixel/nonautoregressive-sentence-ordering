from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import logging
import os
import json
import six
import copy
from io import open
from random import shuffle
import numpy as np
import itertools
from .file_utils import cached_path, is_tf_available, is_torch_available
from transformers import AlbertTokenizer, BertTokenizer, BartTokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

if is_tf_available():
    import tensorflow as tf
if is_torch_available():
    import torch

logger = logging.getLogger(__name__)

class tokenpretrained():

    def encode_plus(self,
                    text,
                    add_special_tokens=False,
                    max_length=None,
                    stride=0,
                    truncation_strategy='longest_first',
                    return_tensors=None):
        first_ids = text[0]

        return self.prepare_for_model(first_ids,
                                      max_length=max_length,
                                      add_special_tokens=add_special_tokens,
                                      stride=stride,
                                      truncation_strategy=truncation_strategy,
                                      return_tensors=return_tensors)

    def pairs_generator(self, lenth):
        '''Generate the combinations of sentence pairs

            Args:
                lenth (int): the length of the sentence

            Returns:
                combs (list) : all combination of the index pairs in the passage
                num_combs (int) : the total number of all combs
        '''
        indices = list(range(lenth))
        combs_one_side = list(itertools.combinations(indices, 2))
        combs_one_side = [[x1, x2] for x1,x2 in combs_one_side]
        combs_other_side = [[x2, x1] for x1,x2 in combs_one_side]
        combs = combs_one_side + combs_other_side
        return combs, len(combs)


    def prepare_for_model(self, ids, max_length=None, add_special_tokens=False, stride=0,
                          truncation_strategy='longest_first', return_tensors=None):
        original_src = ids
        order = list(range(original_src.count('<eos>') + 1))
        passage_length = len(order)
        sents = original_src.strip().split('<eos>')

        # shuffle
        base_index = list(range(passage_length))
        shuffled_index = base_index
        shuffle(shuffled_index)
        ground_truth = list(np.argsort(shuffled_index))
        max_sent_len = 0

        all_para_ids = []
        maskedids = []
        sents_ids = []
        s_indexs = []
        for id in range(len(sents)):
            sent = sents[shuffled_index[id]]
            inputid_sent = tokenizer(sent, truncation=True, max_length=25)['input_ids']
            sents_ids = sents_ids + inputid_sent

        maskid = [1] * len(sents_ids)
        s_index = [i for i, x in enumerate(sents_ids) if x == 0]
        all_para_ids.append(sents_ids)
        maskedids.append(maskid)
        s_indexs.append(s_index)

        encoded_inputs = {}
        encoded_inputs['shuffled_index'] = shuffled_index
        encoded_inputs['ground_truth'] = ground_truth
        encoded_inputs['passage_length'] = passage_length
        encoded_inputs['para_ids'] = all_para_ids
        encoded_inputs['maskedids'] = maskedids
        encoded_inputs['s_indexs'] = s_indexs
        encoded_inputs['sequence_length'] = len(maskid)
        return encoded_inputs
