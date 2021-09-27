import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from sklearn.model_selection import train_test_split

PATH_TO_RAW_ALL_DATASET = "./data/extracted_sentence.csv"
PATH_TO_RAW_TEST_DATASET = "./data/test.csv"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SPACE = " "


def generate_train_test_dataset(path=PATH_TO_RAW_TEST_DATASET, test_size=0.2) -> (
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
    word2idx[START_TOKEN] = 1
    idx2word[1] = START_TOKEN
    word2idx[END_TOKEN] = 2
    idx2word[2] = END_TOKEN
    for i, token in enumerate(vocab):
        word2idx[token] = i + 3
        idx2word[i + 3] = token
    word2idx = word2idx
    idx2word = idx2word
    return word2idx, idx2word, vocab


def generate_embedding_matrix_from_glove(path_to_glove_file: str, vocab: set):
    """
    :param path_to_glove_file: str
    :param vocab: set
    :return:
    """
    pass


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

    target = []
    for y in y_train.tolist():
        temp = []
        for token in y.split():
            temp.append(word2idx[token])
        target.append(temp)

    max_len_input, max_len_output = max(len(i) for i in input_data), max(len(t) for t in target)
    input_data = pad_sequences(input_data, maxlen=max_len_input, padding="post", value=0)
    target = pad_sequences(target, maxlen=max_len_output, padding="post", value=0)
    return input_data, target


def convert_idx_to_sentences(X: np.ndarray, idx2word: dict) -> []:
    """
    :param X: np.ndarray
    :param idx2word: dict
    :return:
    """
    token_list = []
    for idx in X:
        if idx in idx2word.keys():
            token_list.append(idx2word.get(idx))
        else:
            token_list.append(UNK_TOKEN)
    return " ".join(token_list)


def test_main():
    X_train, X_test, y_train, y_test = generate_train_test_dataset()
    word2idx, idx2word, vocab = generate_vocab_dict()
    X, y = generate_train_dataset_for_model(X_train=X_train, y_train=y_train, word2idx=word2idx)
    for i, t in zip(X, y):
        print("Input Seq: {}".format(convert_idx_to_sentences(i, idx2word)))
        print("Target Seq: {}".format(convert_idx_to_sentences(t, idx2word)))


if __name__ == "__main__":
    test_main()
