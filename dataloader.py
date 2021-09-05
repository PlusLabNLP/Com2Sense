"""
Preprocessing Commonsense Datasets
"""
import os
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
        """
        Parses raw dataset file (jsonl). \n
        The complementary sentences are unpaired and treated as independent samples.

        Input:
            [
                {_id: _, 'sent_1': ___, 'label_1': _, 'sent_2': ___, 'label_2': _},
                ...
                {_id: _, 'sent_1': ___, 'label_1': _, 'sent_2': ___, 'label_2': _}
            ]

        Output:
            [
                {_id: _, 'sent_1': ___, 'label_1': 1/0},
                {_id: _, 'sent_2': ___, 'label_2': 1/0},
                ...
                {_id: _, 'sent_1': ___, 'label_1': 1/0},
                {_id: _, 'sent_2': ___, 'label_2': 1/0}
            ]

        :param str data_dir: path to dataset dir
        :returns: sentence, label
        :rtype: list[dict]
        """
        path = os.path.join(data_dir, f'{self.split}.json')

        # Read data
        df = pd.read_json(path)

        # Map labels
        label2int = {'True': 1, 'False': 0}

        if self.split != "test":
            df['label_1'] = df['label_1'].apply(lambda l: label2int[l])
            df['label_2'] = df['label_2'].apply(lambda l: label2int[l])
        else:
            df['label_1'] = -1
            df['label_2'] = -1

        raw_data = df.to_dict(orient='records')

        # add index for pairs
        # TODO remove later
        if self.split == "train":
            for i, pair in enumerate(raw_data):
                pair['_id'] = i
        else:
            # update: Use the database ID
            for i, pair in enumerate(raw_data):
                pair['_id'] = pair['id']

        data = []
        for pair in raw_data:
            sample_1 = dict(_id=pair['_id'], text=pair['sent_1'], label=pair['label_1'])
            sample_2 = dict(_id=pair['_id'], text=pair['sent_2'], label=pair['label_2'])
            data.extend([sample_1, sample_2])

        if self.split == 'train':
            random.seed(0)
            random.shuffle(data)
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
