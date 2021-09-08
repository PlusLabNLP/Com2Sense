"""
Preprocessing Commonsense Datasets
"""
import os
import json
import torch
import random
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Base class for Datasets

    Complementary Commonsense Benchmark

    [True]  It's more comfortable to sleep on a mattress than the floor.
    [False] It's more comfortable to sleep on the floor than a mattress.

    """

    def __init__(self, split, tokenizer, max_seq_len=128, text2text=True, uniqa=False, is_leaderboard=False):
        """
        Processes raw dataset

        :param str split: train/dev/test; (selects `dev` if no `test`)
        :param str tokenizer: tokenizer name (e.g. 'roberta-base', 't5-3b', etc.)
        :param int max_seq_len: tokenized sequence length (padded)
        :param bool text2text: parse dataset in T5 format.
        :param bool uniqa: format dataset in unifiedQA format
        """
        self.split = split
        self.max_seq_len = max_seq_len
        self.text2text = text2text
        self.tok_name = tokenizer
        self.uniqa = uniqa
        self.is_leaderboard = is_leaderboard

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tok_name)

        # Read dataset
        data_dir = self._get_path('com2sense')
        # Process dataset (in subclass)
        self.data = self._preprocess(data_dir)

    def _preprocess(self, data_dir):
        data_path = os.path.join(data_dir, f'{self.split}.json')
        with open(data_path, 'r') as f:
            data_file = json.load(f)

        # we use pair id to get pairwise acc
        pair_id_path = os.path.join(data_dir, f'pair_id_{self.split}.json')
        with open(pair_id_path, 'r') as f:
            data_ids = json.load(f)

        data_df = pd.DataFrame(data_file)
        pairs_map = pd.DataFrame.from_dict(data_ids, orient='index').reset_index()
        pairs_map.columns = ['id1', 'id2']
        joined = pd.merge(data_df, pairs_map, left_on='id', right_on='id1')
        joined_final = pd.merge(joined, data_df, left_on='id2', right_on='id')
        joined_final = joined_final.drop(columns=['id_x', 'id_y', 'domain_y', 'scenario_y', 'numeracy_y'])
        joined_final = joined_final.rename(
            columns={'sent_x': 'sent1', 'label_x': 'label1', 'domain_x': 'domain', 'scenario_x': 'scenario',
                     'numeracy_x': 'numeracy', 'sent_y': 'sent2', 'label_y': 'label2'})
        picked_ids = []
        df = joined_final.to_dict(orient='records')
        data = []
        label_to_int = {'False': 0, 'True': 1}

        for d in df:
            if d['id1'] not in picked_ids:
                picked_ids.extend([d['id1'], d['id2']])
            else:
                continue
            if self.split == 'test':
                sample1 = dict(_id=d['id1'], text=d['sent1'], label=-1)
                sample2 = dict(_id=d['id2'], text=d['sent2'], label=-1)
            else:
                sample1 = dict(_id=d['id1'], text=d['sent1'], label=label_to_int[d['label1']])
                sample2 = dict(_id=d['id2'], text=d['sent2'], label=label_to_int[d['label2']])
            data.append(sample1)
            data.append(sample2)

        if self.split == 'train':
            random.seed(0)
            random.shuffle(data)

        # print(data[:10])
        return data

    def __len__(self):
        return len(self.data)

    def get_tokenizer(self):
        return self.tokenizer

    @staticmethod
    def _get_path(name):
        """Relative paths"""

        paths = {'com2sense': './data'}

        return paths[name]

    def get_classname(self):
        return self.__class__.__name__

    @staticmethod
    def _prepare_text2text(record):
        """
        Input:
            {'text': __, 'label': 1/0}

        Output:
            text: 'c2s sentence: __' \n
            label: 'true' or 'false'

        :returns: text, label
        :rtype: tuple[str]
        """
        input_text = record['text']
        answer = 'true' if record['label'] else 'false'

        # Text-to-Text
        text = f'com2sense sentence: {input_text} </s>'
        label = f'{answer} </s>'

        return text, label

    def max_len_tokenized(self):
        """
        Max tokenized sequence length, assuming text-to-text format
        TODO: Revise it
        """
        return max([len(self.tokenizer.encode(''.join(d.values()))) for d in self.data])

    def __getitem__(self, idx):
        record = self.data[idx]

        if self.text2text:
            # Format input & label
            text, label = self._prepare_text2text(record)

            if self.uniqa:
                text = text.split(':')[1][1:]
                text = 'Is the following sentence correct?\n' + text
                label = label.replace('false', 'no')
                label = label.replace('true', 'yes')

            target_len = 2
            # Tokenize
            input_encoded = self.tokenizer.encode_plus(text=text,
                                                       add_special_tokens=False,
                                                       padding='max_length',
                                                       max_length=self.max_seq_len,
                                                       truncation=True,
                                                       return_attention_mask=True)

            target_encoded = self.tokenizer.encode_plus(text=label,
                                                        add_special_tokens=False,
                                                        padding='max_length',
                                                        max_length=target_len,
                                                        return_attention_mask=True)

            input_token_ids = torch.tensor(input_encoded['input_ids'])
            input_attn_mask = torch.tensor(input_encoded['attention_mask'])

            target_token_ids = torch.tensor(target_encoded['input_ids'])
            target_attn_mask = torch.tensor(target_encoded['attention_mask'])

            # Output
            sample = {'input_tokens': input_token_ids,
                      'input_attn_mask': input_attn_mask,
                      'target_tokens': target_token_ids,
                      'target_attn_mask': target_attn_mask}
        else:

            text, label = record['text'], record['label']

            cls = self.tokenizer.cls_token

            text = f'{cls} {text}'

            tokens = self.tokenizer(text=text,
                                    padding='max_length',
                                    max_length=self.max_seq_len,
                                    add_special_tokens=False,
                                    truncation=True,
                                    return_attention_mask=True)

            token_ids = torch.tensor(tokens['input_ids'])
            attn_mask = torch.tensor(tokens['attention_mask'])

            # Output
            sample = {'tokens': token_ids,
                      'attn_mask': attn_mask,
                      'label': label}

            if self.is_leaderboard:
                sample['_id'] = record['_id']

        return sample
