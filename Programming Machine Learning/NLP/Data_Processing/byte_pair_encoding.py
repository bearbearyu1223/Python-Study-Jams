"""
Use byte pair encoding (BPE) to learn a variable-length encoding of the vocabulary in a text.
"""

import re
import inspect
from typing import Union
from collections import defaultdict, OrderedDict

"""
Algorithm: 
    1. Compute the frequencies of all words in the training corpus
    2. Start with vocabulary that consists from singleton symbols from training corpus
    3. To get vocabulary of n merges, iterate n times:
        a. Get the most frequent pair of symbols in the training data
        b. Add the pair into list of merges 
        c. Add the merged symbol into vocabulary
"""
train_data = {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3}


def _print_debug_info(o: Union[int, str, list, dict, tuple]):
    print("DEBUG :: {}".format(inspect.currentframe().f_code.co_name))
    if isinstance(o, dict):
        for k, v in o.items():
            print("{} : {}".format(k, v).rjust(20))


def get_stats(vocab: dict, debug=False) -> defaultdict:
    """
    compute frequencies of adjacent paris of symbols
    :param vocab:
    :param debug:
    :return:
    """
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    pairs = OrderedDict(sorted(pairs.items(), key=lambda x: x[1]))
    if debug:
        _print_debug_info(o=pairs)
    return pairs


def merge_vocab(pair: str, v_in: dict):
    v_out = {}

    # Return string with all non-alphanumerics back slashed;
    # for example print(re.escape('www.stackoverflow.com')) => www\.stackoverflow\.com
    bigram = re.escape(" ".join(pair))

    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


if __name__ == "__main__":
    bpe_codes = {}
    bpe_codes_reverse = {}

    num_merges = 10

    for i in range(num_merges):
        print("### Iteration {}".format(i + 1))
        pairs = get_stats(train_data)
        best = max(pairs, key=pairs.get)
        train_data = merge_vocab(best, train_data)

        bpe_codes[best] = i
        bpe_codes_reverse[best[0] + best[1]] = best

        print("new merge: {}".format(best))
        print("train data: {}".format(train_data))
