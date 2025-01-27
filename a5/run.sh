#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en \
        --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json --cuda
elif [ "$1" = "test_topk" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_topk.txt
    CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/test_outputs_topk.txt --cuda
elif [ "$1" = "test_dpp" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_dpp.txt
    CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/test_outputs_dpp.txt --cuda --dpp
elif [ "$1" = "test_topk_local" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_topk.txt
    python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/test_outputs_topk.txt
elif [ "$1" = "test_topk_local_small" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_topk_small.txt
    python run.py decode model.bin ./en_es_data/test_small.es ./en_es_data/test_small.en outputs/test_outputs_topk_small.txt
elif [ "$1" = "test_topk_local_medium" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_topk_medium.txt
    python run.py decode model.bin ./en_es_data/test_medium.es ./en_es_data/test_medium.en outputs/test_outputs_topk_medium.txt
elif [ "$1" = "test_topk_local_tiny" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_topk.txt
    python run.py decode model.bin ./en_es_data/test_tiny.es ./en_es_data/test_tiny.en outputs/test_outputs_topk_tiny.txt
elif [ "$1" = "test_dpp_local" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_dpp.txt
    python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/test_outputs_dpp.txt --dpp
elif [ "$1" = "test_dpp_local_small" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_dpp_small.txt
    python run.py decode model.bin ./en_es_data/test_small.es ./en_es_data/test_small.en outputs/test_outputs_dpp_small.txt --dpp
elif [ "$1" = "test_dpp_local_medium" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_dpp_medium.txt
    python run.py decode model.bin ./en_es_data/test_medium.es ./en_es_data/test_medium.en outputs/test_outputs_dpp_medium.txt --dpp
elif [ "$1" = "test_dpp_local_tiny" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_dpp_tiny.txt
    python run.py decode model.bin ./en_es_data/test_tiny.es ./en_es_data/test_tiny.en outputs/test_outputs_dpp_tiny.txt --dpp
elif [ "$1" = "train_local_q1" ]; then
	python run.py train --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
        --dev-src=./en_es_data/dev_tiny.es --dev-tgt=./en_es_data/dev_tiny.en --vocab=vocab_tiny_q1.json --batch-size=2 \
        --valid-niter=100 --max-epoch=101 --no-char-decoder
elif [ "$1" = "test_local_q1" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_local_q1.txt
    python run.py decode model.bin ./en_es_data/test_tiny.es ./en_es_data/test_tiny.en outputs/test_outputs_local_q1.txt \
        --no-char-decoder
elif [ "$1" = "train_local_q2" ]; then
	python run.py train --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
        --dev-src=./en_es_data/dev_tiny.es --dev-tgt=./en_es_data/dev_tiny.en --vocab=vocab_tiny_q2.json --batch-size=2 \
        --max-epoch=201 --valid-niter=100
elif [ "$1" = "test_local_q2" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_local_q2.txt
    python run.py decode model.bin ./en_es_data/test_tiny.es ./en_es_data/test_tiny.en outputs/test_outputs_local_q2.txt 
elif [ "$1" = "vocab" ]; then
    python vocab.py --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
        --size=200 --freq-cutoff=1 vocab_tiny_q1.json
    python vocab.py --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
        vocab_tiny_q2.json
	python vocab.py --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en vocab.json
else
	echo "Invalid Option Selected"
fi
