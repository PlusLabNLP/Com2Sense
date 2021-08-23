"""
THIS IS A CLIENT-SIDE SCRIPT,
NOT DEPLOYED ON THE SERVER.
"""

import os
import torch
import argparse
import requests
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

os.environ['no_proxy'] = '127.0.0.1,localhost'
URL = 'http://127.0.0.1:5000/eval'


@torch.no_grad()
def compute_eval_metrics(model, dataloader, device, size, tokenizer, text2text = False, is_pairwise=False, is_test=False, parallel = False):
    """
    For the given model, computes accuracy & loss on validation/test set.

    :param model: model to evaluate
    :param dataloader: validation/test set dataloader
    :param device: cuda/cpu device where the model resides
    :param size: no. of samples (subset) to use
    :param tokenizer: tokenizer used by the dataloader
    :param is_pairwise: compute the pairwise accuracy
    :param is_test: if set, will return (input, ground-truth & prediction) info under 'meta'
    :return: metrics {'loss', 'accuracy', 'pairwise', 'meta'}
    :rtype: dict
    """
    model.eval()

    # Store predicted & ground-truth labels
    _ids = []
    preds = []

    total_samples = 0
    # Evaluate on mini-batches
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        _id = batch['_id']

        # Forward Pass
        pred = model.inference(batch)       # e.g. [1] or [0]

        # pred_label_list.append({'_id': _id, 'pred_1': pred, 'pred_2': ??})
        preds.append(pred)

        total_samples += dataloader.batch_size

        if total_samples >= size:
            break

    dataset = dataloader.dataset
    _ids = dataset.data['_id']

    output = []

    # Regrouping single sample predictions to pairs
    for i in range(0, len(preds), 2):
        output.append({'_id': _ids[i], 'pred_1': preds[i], 'pred_2': preds[i+1]})

    return preds


def evaluate(model):
    # Dataloader
    dataset = BaseDataset(args.test_file, tokenizer=args.model, max_seq_len=args.seq_len, text2text=text2text,
                          uniqa=uniqa)
    datasets = dataset.concat(dataset_names)

    loader = DataLoader(datasets, batch_size=1, num_workers=args.num_workers)

    tokenizer = dataset.get_tokenizer()

    model.eval()
    model.to(device)

    # TODO: Loading checkpoints for models trained with DataParallel()

    # Load model weights
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    data_len = datasets.__len__()
    print('Total Samples: {}'.format(data_len))

    is_pairwise = 'com2sense' in dataset_names

    # Inference
    metrics = compute_eval_metrics(model, loader, device, data_len, tokenizer, text2text, is_pairwise=is_pairwise,
                                   is_test=True, parallel=args.data_parallel)

    df = pd.DataFrame(metrics['meta'])
    df.to_csv(args.pred_file)

    print(f'Results for model {args.model}')
    print(f'Results evaluated on file {args.test_file}')
    print('Sentence Accuracy: {:.4f}'.format(metrics['accuracy']))
    if is_pairwise:
        print('Pairwise Accuracy: {:.4f}'.format(metrics['pair_acc']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LeaderBoard')

    parser.add_argument('--mode',       type=str, help='eval or submit mode', required=True, choices=['eval', 'submit'])
    parser.add_argument('--ckpt',       type=str, help='path to model checkpoint (.pth)', required=True)
    parser.add_argument('--num_cls',    type=int, help='number of class', default=2)

    # Parse Args
    args = parser.parse_args()

    # ----- MODIFY THIS --------
    model = Transformer(args.model, args.num_cls, text2text, num_layers=args.num_layers)
    # ---------------------------

    predictions = evaluate(model)     # <<<---  PREDICTIONS: pred = {'id': _, 'pred_1': 'True', 'pred_2': 'False'}


    # print(compute_metrics(predictions))
    response = requests.post(url, json=predictions)
    print(response.text)
