#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        
        self.vocab_size = len(target_vocab.char2id)
        self.char_embedding_size = char_embedding_size
        self.charDecoder = nn.LSTM(input_size=char_embedding_size, hidden_size=hidden_size, )
        self.char_output_projection = nn.Linear(hidden_size, self.vocab_size)
        self.target_vocab = target_vocab
        self.decoderCharEmb = nn.Embedding(self.vocab_size, char_embedding_size, padding_idx=target_vocab.char2id['<pad>'])
        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.

        char_embeds = self.decoderCharEmb(input)

        output, dec_hidden = self.charDecoder(char_embeds, dec_hidden)
        scores = self.char_output_projection(output)
        return scores, dec_hidden

        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        """
        loss = nn.CrossEntropyLoss()
        scores, dec_hidden = self.forward(char_sequence, dec_hidden)
        sum_losses = 0
        for i in range(char_sequence.shape[1]):
            output = loss(scores[:,i,:], char_sequence[:,i])
            sum_losses += output
        return sum_losses
        """
        loss = nn.CrossEntropyLoss(reduction='sum', ignore_index=self.target_vocab.char2id['<pad>'])
        scores, dec_hidden = self.forward(char_sequence[0:-1,:], dec_hidden)
        scores = scores.view((-1, self.vocab_size))
        char_sequence_flattened = char_sequence[1:].flatten()
        output = loss(scores, char_sequence_flattened)
        return output


        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        batch_size = initialStates[0].shape[1]
        chars = torch.zeros((batch_size, max_length+1), device=device)
        current_chars = torch.tensor([self.target_vocab.start_of_word]*batch_size, device=device).unsqueeze(0)
        dec_hidden = initialStates
        chars[:,0] = current_chars
        for t in range(max_length):
            scores, dec_hidden = self.forward(current_chars, dec_hidden)
            scores = scores.squeeze(0)
            current_chars = torch.argmax(scores, dim=1)
            chars[:,t+1] = current_chars
            current_chars = current_chars.unsqueeze(0)
        decoded_words = []
        for i in range(batch_size):
            current_word = ""
            for j in range(1,max_length+1):
                c_id = chars[i, j]
                if c_id == self.target_vocab.end_of_word:
                    break
                c = self.target_vocab.id2char[c_id.item()]
                current_word += c
            decoded_words.append(current_word)
        return decoded_words
        ### END YOUR CODE

