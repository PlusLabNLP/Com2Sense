import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report


@torch.no_grad()
def compute_eval_metrics(model, dataloader, device, size, tokenizer, text2text=False, is_pairwise=False, is_test=False,
                         parallel=False):
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
    input_decoded = []
    predicted = []
    ground_truth = []
    loss = []

    def decode(token_ids):
        return tokenizer.decode(token_ids, skip_special_tokens=True)

    total_samples = 0
    # Evaluate on mini-batches
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        # T5 inference
        if text2text:
            # Forward Pass (predict)
            if parallel:
                label_pred = model.module.generate(input_ids=batch['input_tokens'],
                                                   attention_mask=batch['input_attn_mask'],
                                                   max_length=2)
            else:
                label_pred = model.generate(input_ids=batch['input_tokens'],
                                            attention_mask=batch['input_attn_mask'],
                                            max_length=2)
            label_pred = [decode(x).strip() for x in label_pred]

            label_gt = batch['target_tokens']
            label_gt = [decode(x).strip() for x in label_gt]

            input_decoded += [decode(x) for x in batch['input_tokens']]

            # Forward Pass (loss)
            out = model(batch)
            loss.append(out[0])

        # Others
        else:
            # Forward Pass
            label_logits = model(batch)
            label_gt = batch['label']
            label_pred = torch.argmax(label_logits, dim=1)

            input_decoded += [decode(x) for x in batch['tokens']]

            # Loss
            loss.append(F.cross_entropy(label_logits, label_gt, reduction='mean'))

            label_pred = label_pred.detach().cpu().tolist()
            label_gt = label_gt.detach().cpu().tolist()

        # Append batch; list.extend()
        predicted += label_pred
        ground_truth += label_gt

        total_samples += dataloader.batch_size

        if total_samples >= size:
            break

    # Compute metrics
    accuracy = 100 * accuracy_score(ground_truth, predicted)
    pair_acc = 100 * _pairwise_acc(ground_truth, predicted) if is_pairwise else None

    loss = torch.tensor(loss).mean()

    metrics = {'loss': loss,
               'accuracy': accuracy,
               'pair_acc': pair_acc}

    if is_test:
        metrics['meta'] = {'input': input_decoded,
                           'prediction': predicted,
                           'ground_truth': ground_truth}
    return metrics


def _pairwise_acc(y_gt, y_pred):
    assert len(y_gt) == len(y_pred) and len(y_gt) % 2 == 0, 'Invalid Inputs for Pairwise setup'

    res = [y_gt[i] == y_pred[i] for i in range(len(y_gt))]

    # Group by sentence
    res1 = res[0::2]
    res2 = res[1::2]

    pair_acc = np.logical_and(res1, res2).mean()

    return pair_acc


# ---------------------------------------------------------------------------
def setup_logger(parser, log_dir, file_name='train_log.txt'):
    """
    Generates log file and writes the executed python flags for the current run,
    along with the training log (printed to console). \n

    This is helpful in maintaining experiment logs (with arguments). \n

    While resuming training, the new output log is simply appended to the previously created train log file.

    :param parser: argument parser object
    :param log_dir: file path (to create)
    :param file_name: log file name
    :return: train log file
    """
    log_file_path = os.path.join(log_dir, file_name)

    log_file = open(log_file_path, 'a+')

    # python3 file_name.py
    log_file.write('python3 ' + sys.argv[0] + '\n')

    # Add all the arguments (key value)
    args = parser.parse_args()

    for key, value in vars(args).items():
        # write to train log file
        log_file.write('--' + key + ' ' + str(value) + '\n')

    log_file.write('\n\n')
    log_file.flush()

    return log_file


def print_log(msg, log_file):
    """
    :param str msg: Message to be printed & logged
    :param file log_file: log file
    """
    log_file.write(msg + '\n')
    log_file.flush()

    print(msg)


def csv2list(v, cast=str):
    assert type(v) == str, 'Converts: comma-separated string --> list of strings'
    return [cast(s.strip()) for s in v.split(',')]


def str2bool(v):
    v = v.lower()
    assert v in ['true', 'false', 't', 'f', '1', '0'], 'Option requires: "true" or "false"'
    return v in ['true', 't', '1']


def _shuffle(lst):
    np.random.seed(0)
    np.random.shuffle(lst)

    return lst


def train_val_split(data, train_ratio=0.6, dev_ratio=0.2, test_ratio=0.2):
    # Shuffle & Split data
    _shuffle(data)
    split_idx = int(len(data) * train_ratio)

    # train set
    data_train = data[:split_idx]

    # validation set
    rest = data[split_idx:]
    dev_split = int(dev_ratio * len(data))
    data_val = rest[:dev_split]
    rest = rest[dev_split:]
    test_split = int(test_ratio * len(data))
    data_test = rest[:test_split]
    return data_train, data_val, data_test
