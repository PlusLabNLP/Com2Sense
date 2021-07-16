#!/usr/bin/env bash

# ** RoBerta-large || Com2Sense **
python3 main.py --mode train \
--expt_dir ./results_log/com2sense \
--expt_name roberta_large \
--model roberta-large \
--dataset com2sense \
--run bs_16 \
--batch_size 16 \
--seq_len 128


# ** T5-large || Com2Sense **
python3 main.py \
--mode train \
--expt_dir ./results_log/_multi_data/wgd_cqa \
--expt_name t5_large \
--model t5-large \
--dataset com2sense  \
--run bs_8 \
--batch_size 8 \
--seq_len 128 \
--use_amp F
