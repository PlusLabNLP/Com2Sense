"""
THIS IS A CLIENT-SIDE SCRIPT,
NOT DEPLOYED ON THE SERVER.
"""

import os
import yaml
import torch
import argparse
import requests
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import Transformer
from dataloader import BaseDataset
from utils import csv2list, compute_eval_metrics

os.environ['no_proxy'] = '127.0.0.1, localhost'
EVAL_URL = 'http://127.0.0.1:5000/eval'
SUBMIT_URL = 'http://127.0.0.1:5000/submit'


def evaluate(args, model):
    # Multi-GPU
    device_ids = csv2list(args.gpu_ids, int)
    print('Selected GPUs: {}'.format(device_ids))

    # Device for loading dataset (batches)
    device = torch.device(device_ids[0])

    # Text-to-Text
    text2text = ('t5' in args.model)
    uniqa = ('unified' in args.model)

    # Dataloader
    dataset = BaseDataset('test', tokenizer=args.model, max_seq_len=args.seq_len, text2text=text2text,
                          uniqa=uniqa)

    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers)

    # Load model checkpoint file (if specified)
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    # Load model & optimizer
    model.load_weights(checkpoint)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])

    data_len = dataset.__len__()
    print('Total Samples: {}'.format(data_len))

    # Inference
    model.eval()
    model.to(device)

    # Store predicted & ground-truth labels
    _ids = []
    preds = []

    total_samples = 0
    # Evaluate on mini-batches
    for batch in tqdm(loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward Pass
        pred = model.inference(batch)       # e.g. [1] or [0]

        preds.append(pred)

        total_samples += loader.batch_size

        if total_samples >= data_len:
            break

    dataset = loader.dataset
    _ids = [record['_id'] for record in dataset.data]

    output = []

    # Regrouping single sample predictions to pairs
    # use the first id as the pair id
    for i in range(0, len(preds), 2):
        # PREDICTIONS: pred = {'id': _, 'pred_1': 'True', 'pred_2': 'False'}
        pred_1 = 'True' if preds[i][0] == 1 else 'False'
        pred_2 = 'True' if preds[i+1][0] == 1 else 'False'
        output.append({'id': _ids[i], 'pred_1': pred_1, 'pred_2': pred_2})

    return output, model_size


def main():
    parser = argparse.ArgumentParser(description='LeaderBoard')

    parser.add_argument('--mode',       type=str, help='eval or submit mode', required=True, choices=['eval', 'submit'])
    parser.add_argument('--user_info', type=str, help='user info yaml file', default='./submit.yaml')

    # Model params
    parser.add_argument('--model', type=str, help='transformer model (e.g. roberta-base)', required=True)
    parser.add_argument('--ckpt',       type=str, help='path to model checkpoint (.pth)', required=True)

    parser.add_argument('--seq_len', type=int, help='tokenized input sequence length', default=256)
    parser.add_argument('--num_cls',    type=int, help='number of class', default=2)
    parser.add_argument('--num_layers', type=int,
                        help='Number of hidden layers in transformers (default number if not provided)', default=-1)

    # Data params
    parser.add_argument('--pred_file', type=str, help='address of prediction csv file, for "test" mode',
                        default='results.csv')

    # GPU params
    parser.add_argument('--gpu_ids', type=str, help='GPU IDs (0,1,2,..) seperated by comma', default='0')
    parser.add_argument('-data_parallel',
                        help='Whether to use nn.dataparallel (currently available for BERT-based models)',
                        action='store_true')

    # Misc params
    parser.add_argument('--num_workers', type=int, help='number of worker threads for Dataloader', default=1)

    # Parse Args
    args = parser.parse_args()

    text2text = ('t5' in args.model)

    # User could change the model here
    model = Transformer(args.model, args.num_cls, text2text, num_layers=args.num_layers)

    predictions, model_size = evaluate(args, model)

    if args.mode == 'eval':
        response = requests.post(EVAL_URL, json=predictions)
    else:
        with open(args.user_info, 'r') as stream:
            try:
                user_info = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print(e)
        display_model_size = f"{round(float(model_size) / 1e6, 2)}M"
        user_info['model_size'] = display_model_size
        print(user_info)

        post_json = {"predictions": predictions, "user_info": user_info}
        response = requests.post(SUBMIT_URL, json=post_json)
    print(response.text)


if __name__ == '__main__':
    main()
