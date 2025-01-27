#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json --cuda
elif [ "$1" = "test_topk" ]; then
        CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/test_outputs.txt --cuda
elif [ "$1" = "test_dpp" ]; then
        CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/test_outputs_dpp.txt --cuda --dpp
elif [ "$1" = "test_topk_local" ]; then
        python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/test_outputs.txt
elif [ "$1" = "test_dpp_local" ]; then
        python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/test_outputs_dpp.txt --dpp
elif [ "$1" = "test_dpp_local_tiny" ]; then
        python run.py decode model.bin ../a5/en_es_data/test_tiny.es ../a5/en_es_data/test_tiny.en outputs/test_outputs_dpp_tiny.txt --dpp
elif [ "$1" = "train_local" ]; then
	python run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json
elif [ "$1" = "test_local" ]; then
    python run.py decode model.bin ./en_es_data/test_dpp.es ./en_es_data/test_dpp.en outputs/test_outputs_dpp.txt
elif [ "$1" = "vocab" ]; then
	python vocab.py --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en vocab.json
else
	echo "Invalid Option Selected"
fi
