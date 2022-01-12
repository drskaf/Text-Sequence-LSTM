# load in data
import helper
data_dir = 'path to .txt file'
text = helper.load_data(data_dir)

# explore the data
view_line_range = (0, 10)
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))

lines = text.split('\n')
print('Number of lines: {}'.format(len(lines)))
word_count_line = [len(line.split()) for line in lines]
print('Average number of words in each line: {}'.format(np.average(word_count_line)))

print()
print('The lines {} to {}:'.format(*view_line_range))
print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))

#lookup table
import problem_unittests as tests
from collections import Counter

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    
    all_text = ' '.join(text)
    words = all_text.split()
    words_count = Counter(words)
    sorted_vocab = sorted(words_count, key=words_count.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
    # return tuple
    return (vocab_to_int, int_to_vocab)
  
tests.test_create_lookup_tables(create_lookup_tables)

#tochanize punctuation
def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
    punct_dic = {'.':"<Period>", ',':"<Comma>",'"':"<Quotation_Mark>",';':"<Semicolon>",'!':"<Exclamation_mark>",'?':"<Question_mark>",'(':"<Left_Parentheses>",')':"<Right_Parenthesis>",'-':"<Dash>",'\n':"<Return>"}    
    return punct_dic

tests.test_tokenize(token_lookup)

# pre-process training data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
