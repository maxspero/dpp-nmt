#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        self.embed_size = embed_size
        self.e_char = 50
        self.char_embeddings = nn.Embedding(len(vocab.char2id), self.e_char, padding_idx=vocab.char2id['<pad>'])
        self.cnn = CNN(self.e_char, embed_size, 5)
        self.highway = Highway(embed_size, embed_size)
        self.dropout = nn.Dropout(p=0.3)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        sentence_len, batch_size, max_word_len = input.shape
        x = input.contiguous().view(sentence_len * batch_size, max_word_len)
        x_emb = self.char_embeddings(x)
        x_reshaped = x_emb.permute(0,2,1)
        x_conv_out = self.cnn(x_reshaped)
        x_highway = self.highway(x_conv_out)
        x_word_emb = self.dropout(x_highway)
        x_out = x_word_emb.view(sentence_len, batch_size, self.embed_size)
        """ debugging prints
        print(input.shape)
        print(x.shape)
        print(x_emb.shape)
        print(x_reshaped.shape)
        print(x_conv_out.shape)
        print(x_highway.shape)
        print(x_word_emb.shape)
        print(x_out.shape)
        """
        return x_out
        ### END YOUR CODE

