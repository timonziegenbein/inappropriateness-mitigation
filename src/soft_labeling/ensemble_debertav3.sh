#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
python ensemble_debertav3.py \
    --repeat 0 \
    --output ../../data/models/binary-debertav3-conservative-no-issue/ \
    --input ../../data/iac2/convinceme.csv \
    --text_col text \
    --checkpoint checkpoint-1800 \
    --model_count 5 \
    --id_col text_id \
    --dataset_name convinceme

python ensemble_debertav3.py \
    --repeat 0 \
    --output ../../data/models/binary-debertav3-conservative-no-issue/ \
    --input ../../data/iac2/createdebate.csv \
    --text_col text \
    --checkpoint checkpoint-1800 \
    --model_count 5 \
    --id_col text_id \
    --dataset_name createdebate 

python ensemble_debertav3.py \
    --repeat 0 \
    --output ../../data/models/binary-debertav3-conservative-no-issue/ \
    --input ../../data/GAQCorpus_split/GAQ.csv \
    --text_col text \
    --checkpoint checkpoint-1800 \
    --model_count 5 \
    --id_col id \
    --dataset_name GAQ
