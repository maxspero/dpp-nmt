#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
run.py: Run Script for Simple NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>

Usage:
    run_comparison.py MODEL_PATH TEST_SOURCE_FILE 
    run_comparison.py MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE 

Options:
    --seed=<int>                            seed [default: 0]
    --beam_size=<int>                       beam size [default: 5]
    --max-decoding-time-step=<int>          max decoding time step [default: 70]
"""
import math
import sys
import pickle
import time


from docopt import docopt
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nmt_model import Hypothesis, NMT
from nmt_model_dpp import DPPNMT
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import read_corpus, batch_iter
from vocab import Vocab, VocabEntry

import torch
import torch.nn.utils

def decode(args: Dict[str, str]):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    @param args (Dict): args from cmd line
    """

    print("load test source sentences from [{}]".format(args['TEST_SOURCE_FILE']), file=sys.stderr)
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        print("load test target sentences from [{}]".format(args['TEST_TARGET_FILE']), file=sys.stderr)
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print("load model from {}".format(args['MODEL_PATH']), file=sys.stderr)
    model = NMT.load(args['MODEL_PATH'])
    model2 = DPPNMT.load(args['MODEL_PATH'])

    beam_search2(
        model, 
        model2,
        test_data_src,
        #int(args['--beam-size']),
        5,
        #int(args['--max-decoding-time-step']), 
        70,
        test_data_tgt,
    )


def beam_search2(model1: NMT, model2: DPPNMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int, test_data_tgt) -> List[List[Hypothesis]]:
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (NMT): NMT Model
    @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    model1.eval()
    model2.eval()

    i = 0;
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            hyp1 = model1.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)
            hyp2 = model2.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)
            ref = test_data_tgt[i][1:-1]
            #print(ref, hyp1[0].value)
            bleu_topk = sentence_bleu(ref, hyp1[0].value)
            bleu_dpp = sentence_bleu(test_data_tgt[i], hyp2[0].value)
            #print(bleu_topk, bleu_dpp)
            if bleu_dpp > bleu_topk:
                print(i)
                print(" ".join(hyp1[0].value))
                print(" ".join(hyp2[0].value))
                print(" ".join(ref))
            i += 1


    return hypotheses


def main():
    """ Main func.
    """
    args = docopt(__doc__)

    # Check pytorch version
    assert(torch.__version__ == "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    # seed the random number generators
    #seed = int(args['--seed'])
    seed = int(0)
    torch.manual_seed(seed)
    np.random.seed(seed * 13 // 7)
    decode(args)


if __name__ == '__main__':
    main()
