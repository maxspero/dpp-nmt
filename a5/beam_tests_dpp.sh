#!/bin/bash

for k in 1 2 3 4 5 10 20 50
do 
    python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/test_outputs_dpp-$k.txt --dpp --beam-size $k
done