from unittest.mock import MagicMock, patch
import numpy as np
import torch


class _TestNN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(_TestNN, self).__init__()
        self.decoder = torch.nn.Linear(input_size, output_size)
        self.forward_called = False
    
    def forward(self, nn_input, hidden):
        self.forward_called = True
        output = self.decoder(nn_input)
        
        return output, hidden


def _print_success_message():
    print('Tests Passed')


class AssertTest(object):
    def __init__(self, params):
        self.assert_param_message = '\n'.join([str(k) + ': ' + str(v) + '' for k, v in params.items()])
    
    def test(self, assert_condition, assert_message):
        assert assert_condition, assert_message + '\n\nUnit Test Function Parameters\n' + self.assert_param_message


def test_create_lookup_tables(create_lookup_tables):
    test_text = '''
        Moe_Szyslak Moe's Tavern Where the elite meet to drink
        Bart_Simpson Eh yeah hello is Mike there Last name Rotch
        Moe_Szyslak Hold on I'll check Mike Rotch Mike Rotch Hey has anybody seen Mike Rotch lately
        Moe_Szyslak Listen you little puke One of these days I'm gonna catch you and I'm gonna carve my name on your back with an ice pick
        Moe_Szyslak Whats the matter Homer You're not your normal effervescent self
        Homer_Simpson I got my problems Moe Give me another one
        Moe_Szyslak Homer hey you should not drink to forget your problems
        Barney_Gumble Yeah you should only drink to enhance your social skills'''
    
    test_text = test_text.lower()
    test_text = test_text.split()
    
    vocab_to_int, int_to_vocab = create_lookup_tables(test_text)
    
    # Check types
    assert isinstance(vocab_to_int, dict),\
        'vocab_to_int is not a dictionary.'
    assert isinstance(int_to_vocab, dict),\
        'int_to_vocab is not a dictionary.'
    
    # Compare lengths of dicts
    assert len(vocab_to_int) == len(int_to_vocab),\
        'Length of vocab_to_int and int_to_vocab don\'t match. ' \
        'vocab_to_int is length {}. int_to_vocab is length {}'.format(len(vocab_to_int), len(int_to_vocab))

    # Make sure the dicts have the same words
    vocab_to_int_word_set = set(vocab_to_int.keys())
    int_to_vocab_word_set = set(int_to_vocab.values())

    assert not (vocab_to_int_word_set - int_to_vocab_word_set),\
    'vocab_to_int and int_to_vocab don\'t have the same words.' \
        '{} found in vocab_to_int, but not in int_to_vocab'.format(vocab_to_int_word_set - int_to_vocab_word_set)
    assert not (int_to_vocab_word_set - vocab_to_int_word_set),\
        'vocab_to_int and int_to_vocab don\'t have the same words.' \
        '{} found in int_to_vocab, but not in vocab_to_int'.format(int_to_vocab_word_set - vocab_to_int_word_set)
    
    # Make sure the dicts have the same word ids
    vocab_to_int_word_id_set = set(vocab_to_int.values())
    int_to_vocab_word_id_set = set(int_to_vocab.keys())
    
    assert not (vocab_to_int_word_id_set - int_to_vocab_word_id_set),\
        'vocab_to_int and int_to_vocab don\'t contain the same word ids.' \
        '{} found in vocab_to_int, but not in int_to_vocab'.format(vocab_to_int_word_id_set - int_to_vocab_word_id_set)
    assert not (int_to_vocab_word_id_set - vocab_to_int_word_id_set),\
        'vocab_to_int and int_to_vocab don\'t contain the same word ids.' \
        '{} found in int_to_vocab, but not in vocab_to_int'.format(int_to_vocab_word_id_set - vocab_to_int_word_id_set)
    
    # Make sure the dicts make the same lookup
    missmatches = [(word, id, id, int_to_vocab[id]) for word, id in vocab_to_int.items() if int_to_vocab[id] != word]
    
    assert not missmatches,\
        'Found {} missmatche(s). First missmatch: vocab_to_int[{}] = {} and int_to_vocab[{}] = {}'.format(len(missmatches),
                                                                                                          *missmatches[0])
    
    assert len(vocab_to_int) > len(set(test_text))/2,\
        'The length of vocab seems too small.  Found a length of {}'.format(len(vocab_to_int))
    
    _print_success_message()
