import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from pprint import pprint

PATH_TO_RAW_ALL_DATASET = "./data/extracted_sentence.csv"
PATH_TO_RAW_TEST_DATASET = "data/test.csv"
PATH_TO_GLOVE_FILE = "./data/glove.6B.100d.txt"
START_TOKEN = "[START]"
END_TOKEN = "[END]"
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
SPACE = " "


def generate_word_based_train_test_dataset(path=PATH_TO_RAW_TEST_DATASET, test_size=0.2) -> (
        np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    :param path: str
    :param test_size: float
    :return:
    """
    input_data = pd.read_csv(path, engine="python")
    df = pd.DataFrame()
    for i in tqdm(range(input_data.shape[0])):
        token_list = input_data.loc[i, 'sent'].split()
        for j in range(1, len(token_list)):
            X = START_TOKEN + SPACE + " ".join(token_list[:j]) + SPACE + END_TOKEN
            y = START_TOKEN + SPACE + " ".join(token_list[j:]) + SPACE + END_TOKEN
            df = df.append({"X": X, "y": y}, ignore_index=True)
    X_train, X_test, y_train, y_test = train_test_split(df["X"], df["y"], test_size=test_size, random_state=42)
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    return X_train, X_test, y_train, y_test


def generate_vocab_dict(path=PATH_TO_RAW_TEST_DATASET) -> (dict, dict, set):
    """
    :param path: str
    :return:
    """
    input_data = pd.read_csv(path, engine="python")
    word2idx = {}
    idx2word = {}
    vocab = set()
    for i in tqdm(range(input_data.shape[0])):
        token_list = input_data.loc[i, "sent"].split(" ")
        vocab.update(token_list)
    vocab = sorted(vocab)
    word2idx[PAD_TOKEN] = 0
    idx2word[0] = PAD_TOKEN
    word2idx[UNK_TOKEN] = 1
    idx2word[1] = UNK_TOKEN
    word2idx[START_TOKEN] = 2
    idx2word[2] = START_TOKEN
    word2idx[END_TOKEN] = 3
    idx2word[3] = END_TOKEN
    for i, token in enumerate(vocab):
        word2idx[token] = i + 4
        idx2word[i + 4] = token
    return word2idx, idx2word, vocab


def generate_embedding_matrix_from_glove(vocab: set, word2idx: dict, path_to_pretrained_embedding=PATH_TO_GLOVE_FILE):
    embedding_index = {}
    with open(path_to_pretrained_embedding) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embedding_index[word] = coefs
    pprint("Found {} word vectors.".format(len(embedding_index)))

    num_tokens = len(vocab) + 4
    embedding_dim = 100
    hits = 0
    misses = 0

    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word2idx.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits = hits + 1
        else:
            misses = misses + 1
    pprint("Convert {} words and miss {} words".format(hits, misses))
    return embedding_matrix


def generate_train_dataset_for_model(X_train: np.ndarray, y_train: np.ndarray, word2idx: dict) -> (
        np.ndarray, np.ndarray):
    """
    :param X_train: np.ndarray
    :param y_train: np.ndarray
    :param word2idx: dict
    :return:
    """
    input_data = []
    for x in X_train.tolist():
        temp = []
        for token in x.split():
            temp.append(word2idx[token])
        input_data.append(temp)

    teacher_data = []
    for y in y_train.tolist():
        temp = []
        for token in y.split():
            temp.append(word2idx[token])
        teacher_data.append(temp)

    max_len_input, max_len_output = max(len(i) for i in input_data), max(len(t) for t in teacher_data)
    input_data = pad_sequences(input_data, maxlen=max_len_input, padding="post", value=0)
    teacher_data = pad_sequences(teacher_data, maxlen=max_len_output, padding="post", value=0)

    target_data = [[teacher_data[n][i + 1] for i in range(len(teacher_data[n]) - 1)] for n in range(len(teacher_data))]
    target_data = pad_sequences(target_data, maxlen=max_len_output, padding="post", value=0)
    target_data = target_data.reshape((target_data.shape[0], target_data.shape[1]))

    p = np.random.permutation(len(input_data))
    input_data = input_data[p]
    teacher_data = teacher_data[p]
    target_data = target_data[p]

    return input_data, teacher_data, target_data, max_len_input, max_len_output


def convert_idx_to_sentences(X: np.ndarray, idx2word: dict) -> []:
    """
    :param X: np.ndarray
    :param idx2word: dict
    :return:
    """
    token_list = []
    for idx in X:
        if idx in idx2word.keys() and idx != 0:
            token_list.append(idx2word.get(idx))
    return " ".join(token_list)


def convert_sentences_to_idx(word2idx: dict, input_sentence: str) -> []:
    """
    :param word2idx: dict
    :param input_sentence: str
    :return:
    """
    token_list = input_sentence.split()
    idx = []
    for token in token_list:
        if token.lower() in word2idx.keys():
            idx.append(word2idx[token.lower()])
        elif token in [START_TOKEN, END_TOKEN, PAD_TOKEN]:
            idx.append(word2idx[token])
        else:
            idx.append(word2idx[UNK_TOKEN])
    return idx


def test_main():
    X_train, X_test, y_train, y_test = generate_word_based_train_test_dataset()
    word2idx, idx2word, vocab = generate_vocab_dict()
    _ = generate_embedding_matrix_from_glove(vocab=vocab, word2idx=word2idx)
    input_data, teacher_data, target_data, max_len_input, max_len_output = generate_train_dataset_for_model(
        X_train=X_train, y_train=y_train,
        word2idx=word2idx)
    for i, t, r in zip(input_data, teacher_data, target_data):
        print("Input Seq: {}".format(convert_idx_to_sentences(i, idx2word)))
        print("Teacher Seq: {}".format(convert_idx_to_sentences(t, idx2word)))
        print("Target Seq: {}".format(convert_idx_to_sentences(r, idx2word)))
    input_sentence = START_TOKEN + SPACE + "How Are You Doing SWEETII" + SPACE + PAD_TOKEN + SPACE + END_TOKEN
    idx = convert_sentences_to_idx(word2idx, input_sentence)
    print("Original Sentence: {}".format(input_sentence))
    print("idx representation: {}".format(idx))
    print("Recovered Sentence: {}".format(convert_idx_to_sentences(np.asarray(idx), idx2word)))


if __name__ == "__main__":
    test_main()
