import torch.nn as nn
import argparse
import pandas as pd
from time import time
from model import Transformer
from dataloader import BaseDataset

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from utils import *
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    parser = argparse.ArgumentParser(description='Commonsense Dataset Dev')

    # Experiment params
    parser.add_argument('--mode', type=str, help='train or test mode', required=True, choices=['train', 'test'])
    parser.add_argument('--expt_dir', type=str, help='root directory to save model & summaries')
    parser.add_argument('--expt_name', type=str, help='expt_dir/expt_name: organize experiments')
    parser.add_argument('--run_name', type=str, help='expt_dir/expt_name/run_name: organize training runs')
    parser.add_argument('--test_file', type=str, default='test',
                        help='The file containing test data to evaluate in test mode.')

    # Model params
    parser.add_argument('--model', type=str, help='transformer model (e.g. roberta-base)', required=True)
    parser.add_argument('--num_layers', type=int,
                        help='Number of hidden layers in transformers (default number if not provided)', default=-1)
    parser.add_argument('--seq_len', type=int, help='tokenized input sequence length', default=256)
    parser.add_argument('--num_cls', type=int, help='model number of classes', default=2)
    parser.add_argument('--ckpt', type=str, help='path to model checkpoint .pth file')

    # Data params
    parser.add_argument('--pred_file', type=str, help='address of prediction csv file, for "test" mode',
                        default='results.csv')
    parser.add_argument('--dataset', type=str, default='com2sense')
    # Training params
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-5)
    parser.add_argument('--epochs', type=int, help='number of epochs', default=100)
    parser.add_argument('--batch_size', type=int, help='batch size', default=8)
    parser.add_argument('--acc_step', type=int, help='gradient accumulation steps', default=1)
    parser.add_argument('--log_interval', type=int, help='interval size for logging training summaries', default=100)
    parser.add_argument('--save_interval', type=int, help='save model after `n` weight update steps', default=30000)
    parser.add_argument('--val_size', type=int, help='validation set size for evaluating metrics, '
                                                     'and it need to be even to get pairwise accuracy', default=2048)

    # GPU params
    parser.add_argument('--gpu_ids', type=str, help='GPU IDs (0,1,2,..) seperated by comma', default='0')
    parser.add_argument('-data_parallel',
                        help='Whether to use nn.dataparallel (currently available for BERT-based models)',
                        action='store_true')
    parser.add_argument('--use_amp', type=str2bool, help='Automatic-Mixed Precision (T/F)', default='T')
    parser.add_argument('-cpu', help='use cpu only (for test)', action='store_true')

    # Misc params
    parser.add_argument('--num_workers', type=int, help='number of worker threads for Dataloader', default=1)

    # Parse Args
    args = parser.parse_args()

    # Dataset list
    dataset_names = csv2list(args.dataset)
    print()

    # Multi-GPU
    device_ids = csv2list(args.gpu_ids, int)
    print('Selected GPUs: {}'.format(device_ids))

    # Device for loading dataset (batches)
    device = torch.device(device_ids[0])
    if args.cpu:
        device = torch.device('cpu')

    # Text-to-Text
    text2text = ('t5' in args.model)
    uniqa = ('unified' in args.model)

    assert not (text2text and args.use_amp == 'T'), 'use_amp should be F when using T5-based models.'
    # Train params
    n_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    accumulation_steps = args.acc_step
    # Todo: Verify the grad-accum code (loss avging seems slightly incorrect)

    # Train
    if args.mode == 'train':
        # Ensure CUDA available for training
        assert torch.cuda.is_available(), 'No CUDA device for training!'

        # Setup train log directory
        log_dir = os.path.join(args.expt_dir, args.expt_name, args.run_name)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # TensorBoard summaries setup  -->  /expt_dir/expt_name/run_name/
        writer = SummaryWriter(log_dir)

        # Train log file
        log_file = setup_logger(parser, log_dir)

        print('Training Log Directory: {}\n'.format(log_dir))

        # Dataset & Dataloader
        dataset = BaseDataset('train', tokenizer=args.model, max_seq_len=args.seq_len, text2text=text2text, uniqa=uniqa)
        train_datasets = ConcatDataset([dataset])

        dataset = BaseDataset('dev', tokenizer=args.model, max_seq_len=args.seq_len, text2text=text2text, uniqa=uniqa)
        val_datasets = ConcatDataset([dataset])

        train_loader = DataLoader(train_datasets, batch_size, shuffle=True, drop_last=True,
                                  num_workers=args.num_workers)
        val_loader = DataLoader(val_datasets, batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)

        # In multi-dataset setups, also track dataset-specific loaders for validation metrics
        val_dataloaders = []
        if len(dataset_names) > 1:
            for val_dset in val_datasets.datasets:
                loader = DataLoader(val_dset, batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)

                val_dataloaders.append(loader)

        # Tokenizer
        tokenizer = dataset.get_tokenizer()

        # Split sizes
        train_size = train_datasets.__len__()
        val_size = val_datasets.__len__()
        log_msg = 'Train: {} \nValidation: {}\n\n'.format(train_size, val_size)

        # Min of the total & subset size
        val_used_size = min(val_size, args.val_size)
        log_msg += 'Validation Accuracy is computed using {} samples. See --val_size\n'.format(val_used_size)

        log_msg += 'No. of Classes: {}\n'.format(args.num_cls)
        print_log(log_msg, log_file)

        # Build Model
        model = Transformer(args.model, args.num_cls, text2text, device_ids, num_layers=args.num_layers)
        if args.data_parallel and not args.ckpt:
            model = nn.DataParallel(model, device_ids=device_ids)
            device = torch.device(f'cuda:{model.device_ids[0]}')

        if not text2text:
            model.to(device)

        model.train()

        # Loss & Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr)
        optimizer.zero_grad()

        scaler = GradScaler(enabled=args.use_amp)

        # Step & Epoch
        start_epoch = 1
        curr_step = 1
        best_val_acc = 0.0

        # Load model checkpoint file (if specified)
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location=device)

            # Load model & optimizer
            model.load_state_dict(checkpoint['model_state_dict'])
            if args.data_parallel:
                model = nn.DataParallel(model, device_ids=device_ids)
                device = torch.device(f'cuda:{model.device_ids[0]}')
            model.to(device)

            curr_step = checkpoint['curr_step']
            start_epoch = checkpoint['epoch']
            prev_loss = checkpoint['loss']

            log_msg = 'Resuming Training...\n'
            log_msg += 'Model successfully loaded from {}\n'.format(args.ckpt)
            log_msg += 'Training loss: {:2f} (from ckpt)\n'.format(prev_loss)

            print_log(log_msg, log_file)

        steps_per_epoch = len(train_loader)
        start_time = time()

        for epoch in range(start_epoch, start_epoch + n_epochs):
            for batch in tqdm(train_loader):
                # Load batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                with autocast(args.use_amp):
                    if text2text:
                        # Forward + Loss
                        output = model(batch)
                        loss = output[0]

                    else:
                        # Forward Pass
                        label_logits = model(batch)
                        label_gt = batch['label']

                        # Compute Loss
                        loss = criterion(label_logits, label_gt)

                if args.data_parallel:
                    loss = loss.mean()
                # Backward Pass
                loss /= accumulation_steps
                scaler.scale(loss).backward()

                if curr_step % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                # Print Results - Loss value & Validation Accuracy
                if curr_step % args.log_interval == 0:
                    # Validation set accuracy
                    if val_datasets:
                        val_metrics = compute_eval_metrics(model, val_loader, device, val_used_size, tokenizer,
                                                           text2text, parallel=args.data_parallel)

                        # Reset the mode to training
                        model.train()

                        log_msg = 'Validation Accuracy: {:.2f} %  || Validation Loss: {:.4f}'.format(
                            val_metrics['accuracy'], val_metrics['loss'])

                        print_log(log_msg, log_file)

                        # Add summaries to TensorBoard
                        writer.add_scalar('Val/Loss', val_metrics['loss'], curr_step)
                        writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], curr_step)

                    # Add summaries to TensorBoard
                    writer.add_scalar('Train/Loss', loss.item(), curr_step)

                    # Compute elapsed & remaining time for training to complete
                    time_elapsed = (time() - start_time) / 3600

                    log_msg = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} | time elapsed: {:.2f}h |'.format(
                        epoch, n_epochs, curr_step, steps_per_epoch, loss.item(), time_elapsed)

                    print_log(log_msg, log_file)

                # Save the model
                if curr_step % args.save_interval == 0:
                    path = os.path.join(log_dir, 'model_' + str(curr_step) + '.pth')

                    state_dict = {'model_state_dict': model.state_dict(),
                                  'curr_step': curr_step, 'loss': loss.item(),
                                  'epoch': epoch, 'val_accuracy': best_val_acc}

                    torch.save(state_dict, path)

                    log_msg = 'Saving the model at the {} step to directory:{}'.format(curr_step, log_dir)
                    print_log(log_msg, log_file)

                curr_step += 1

            # Validation accuracy on the entire set
            if val_datasets:
                log_msg = '-------------------------------------------------------------------------\n'
                val_metrics = compute_eval_metrics(model, val_loader, device, val_size, tokenizer, text2text,
                                                   parallel=args.data_parallel)

                log_msg += '\nAfter {} epoch:\n'.format(epoch)
                log_msg += 'Validation Accuracy: {:.2f} %  || Validation Loss: {:.4f}\n'.format(
                    val_metrics['accuracy'], val_metrics['loss'])

                # For Multi-Dataset setup:
                if len(dataset_names) > 1:
                    # compute validation set metrics on each dataset independently
                    for loader in val_dataloaders:
                        metrics = compute_eval_metrics(model, loader, device, val_size, tokenizer, text2text,
                                                       parallel=args.data_parallel)

                        log_msg += '\n --> {}\n'.format(loader.dataset.get_classname())
                        log_msg += 'Validation Accuracy: {:.2f} %  || Validation Loss: {:.4f}\n'.format(
                            metrics['accuracy'], metrics['loss'])

                # Save best model after every epoch
                if val_metrics["accuracy"] > best_val_acc:
                    best_val_acc = val_metrics["accuracy"]

                    step = '{:.1f}k'.format(curr_step / 1000) if curr_step > 1000 else '{}'.format(curr_step)
                    filename = 'ep_{}_stp_{}_acc_{:.4f}_{}.pth'.format(
                        epoch, step, best_val_acc, args.model.replace('-', '_').replace('/', '_'))

                    path = os.path.join(log_dir, filename)
                    if args.data_parallel:
                        model_state_dict = model.module.state_dict()
                    else:
                        model_state_dict = model.state_dict()
                    state_dict = {'model_state_dict': model_state_dict,
                                  'curr_step': curr_step, 'loss': loss.item(),
                                  'epoch': epoch, 'val_accuracy': best_val_acc}

                    torch.save(state_dict, path)

                    log_msg += "\n** Best Performing Model: {:.2f} ** \nSaving weights at {}\n".format(best_val_acc,
                                                                                                       path)

                log_msg += '-------------------------------------------------------------------------\n\n'
                print_log(log_msg, log_file)

                # Reset the mode to training
                model.train()

        writer.close()
        log_file.close()

    elif args.mode == 'test':

        # Dataloader
        dataset = BaseDataset(args.test_file, tokenizer=args.model, max_seq_len=args.seq_len, text2text=text2text,
                              uniqa=uniqa)

        loader = DataLoader(dataset, batch_size, num_workers=args.num_workers)

        tokenizer = dataset.get_tokenizer()

        model = Transformer(args.model, args.num_cls, text2text, num_layers=args.num_layers)
        model.eval()
        model.to(device)

        # Load model weights
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        data_len = dataset.__len__()
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
    main()
