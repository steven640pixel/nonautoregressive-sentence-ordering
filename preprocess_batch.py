import torch
import numpy as np
import torch
import numpy as np


def preprocess(batch):
    sen_num_dataset = []
    sens_len_dataset = []

    for example in batch:
        sen_num_dataset.append(example['passage_length'])
        sens_len_dataset.append(example['sequence_length'])

    max_sen_num = max(sen_num_dataset)
    max_sents_len = max(sens_len_dataset)

    all_input_ids = []
    all_attention_mask = []
    all_passage_length = []
    all_ground_truth = []
    all_s_index = []

    for inputs in batch:

        input_ids, masked_ids = inputs['para_ids'], inputs['maskedids']
        shuffled_index, ground_truth = inputs['shuffled_index'], inputs[
            'ground_truth']
        passage_length = inputs['passage_length']
        sents_length= inputs['sequence_length']
        s_index = inputs['s_indexs']

        padd_num_sen = max_sen_num - passage_length

        pad_id = 0

        padding_sents_len = max_sents_len - sents_length

        passage_length_new = passage_length

        ground_truth_new = ground_truth + [pad_id] * padd_num_sen

        all_input_ids.append(input_ids[0] + [pad_id] * padding_sents_len)
        all_attention_mask.append(masked_ids[0] + [pad_id] * padding_sents_len)


        all_passage_length.append(passage_length_new)

        all_ground_truth.append(ground_truth_new)
        all_s_index.append(s_index[0] + [pad_id] * padd_num_sen)

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_passage_length = torch.tensor(all_passage_length, dtype=torch.long)
    all_ground_truth = torch.tensor(all_ground_truth, dtype=torch.long)
    all_s_index = torch.tensor(all_s_index, dtype=torch.long)

    new_batch = [all_input_ids, all_attention_mask, all_passage_length,
                 all_ground_truth, all_s_index]

    return new_batch





