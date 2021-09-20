from tqdm import tqdm
import pandas as pd
from pprint import pprint
from typing import List
import tensorflow as tf

PATH_TO_RAW_DATASET = "./data/extracted_sentence.csv"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
PAD_TOKEN = "<pad>"
UNKNOW_TOKEN = "<unk>"


def generate_dataset(corpus: List[str]) -> (List[List], pd.DataFrame):
    output = []
    for p in tqdm(range(len(corpus))):
        line = corpus[p]
        for i in range(1, len(line)):
            data = []
            x_ngram = START_TOKEN + ' ' + line[:i + 1] + ' ' + END_TOKEN
            y_ngram = START_TOKEN + ' ' + line[i + 1:] + ' ' + END_TOKEN
            data.append(x_ngram)
            data.append(y_ngram)
            output.append(data)
    output_df = pd.DataFrame(output, columns=["input", "output"])
    return output, output_df


class LanguageIndex:
    def __init__(self, lang: List[str]):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_index()

    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))
        self.vocab = sorted(self.vocab)
        self.word2idx[PAD_TOKEN] = 0
        self.idx2word[0] = PAD_TOKEN
        self.word2idx[UNKNOW_TOKEN] = 1
        self.idx2word[1] = UNKNOW_TOKEN
        self.word2idx[START_TOKEN] = 2
        self.idx2word[2] = START_TOKEN
        self.word2idx[END_TOKEN] = 3
        self.idx2word[3] = END_TOKEN
        i = 0
        for _, word in enumerate(self.vocab):
            if word not in [START_TOKEN, END_TOKEN, PAD_TOKEN, UNKNOW_TOKEN]:
                self.word2idx[word] = i + 4
                self.idx2word[i + 4] = word
                i = i + 1


def max_len(inputs: List) -> int:
    return max(len(i) for i in inputs)


def load_dataset(sample_data: pd.DataFrame, col="sent") -> (
tf.Tensor, tf.Tensor, LanguageIndex, LanguageIndex, int, int, pd.DataFrame):
    corpus = sample_data[col].values.tolist()
    pairs, df = generate_dataset(corpus)
    out_lang = LanguageIndex(sp for en, sp in pairs)
    in_lang = LanguageIndex(en for en, sp in pairs)
    input_data = [[in_lang.word2idx[s] for s in en.split(' ')] for en, sp in pairs]
    output_data = [[out_lang.word2idx[s] for s in sp.split(' ')] for en, sp in pairs]

    max_length_in, max_length_out = max_len(input_data), max_len(output_data)
    input_data = tf.keras.preprocessing.sequence.pad_sequences(input_data, maxlen=max_length_in, padding="post")
    output_data = tf.keras.preprocessing.sequence.pad_sequences(output_data, maxlen=max_length_out, padding="post")

    return input_data, output_data, in_lang, out_lang, max_length_in, max_length_out, df


def run(path=PATH_TO_RAW_DATASET, test=False):
    if test:
        sample_data = ["here are the deltas, thanks john"]
    else:
        sample_data = pd.read_csv(path, engine="python")
    input_data, output_data, in_lang, out_lang, max_length_in, max_length_out, df = load_dataset(sample_data)
    pprint("First 20 items in the input dict...")
    pprint(list(in_lang.idx2word.items())[:20])
    pprint("First 20 items in the output dict...")
    pprint(list(out_lang.idx2word.items())[:20])
    pprint("First 20 items in the input-output pair...")
    pprint(df.head(20))


if __name__ == "__main__":
    run()
