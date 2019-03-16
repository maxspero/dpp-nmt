#!/bin/bash

for k in 3 4 5 10 20 50 100
do 
    python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/test_outputs_topk-$k.txt --beam-size $k
done
