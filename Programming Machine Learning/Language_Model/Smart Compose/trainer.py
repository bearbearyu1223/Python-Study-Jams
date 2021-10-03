import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import EncoderDecoderModel
from preprocess import generate_word_based_train_test_dataset, generate_train_dataset_for_model, generate_vocab_dict, \
    convert_sentences_to_idx, START_TOKEN, END_TOKEN, SPACE, UNK_TOKEN, generate_embedding_matrix_from_glove
from pprint import pprint
import matplotlib.pyplot as plt
from nltk.translate import bleu_score


def get_txt_from_idx(idx: list, idx2word: dict) -> str:
    sent = ''
    for i in idx:
        if i != 0:
            sent += idx2word[i] + ' '
    return sent


def get_predicted_txt_seq(enc_dec_model: EncoderDecoderModel, output_max_length: int, word2idx: dict, idx2word: dict,
                          input_seq: str) -> str:
    # Get output from encoder
    input_seq = START_TOKEN + SPACE + input_seq + SPACE + END_TOKEN
    input_seq = convert_sentences_to_idx(word2idx, input_sentence=input_seq)
    input_seq = pad_sequences([input_seq], maxlen=max_len_output, padding="post", value=0)
    _, enc_hidden, enc_cell = enc_dec_model.layers[0](input_seq)

    # Boundary case for decoder
    dec_input = tf.expand_dims([word2idx[START_TOKEN]], 1)
    dec_hidden = enc_hidden
    dec_cell = enc_cell

    # Predicted output idx
    # Add <start> token to output idx
    output_seq = [word2idx[START_TOKEN]]

    # The model will start predicting after start token, hence max_length is subtracted by 1
    for i in range(output_max_length - 1):
        # Get prediction from decoder
        outputs = enc_dec_model.layers[1](dec_input, dec_hidden, dec_cell)

        # Extract predicted id from decoder output
        key = np.argmax(outputs.numpy().reshape(-1))

        output_seq.append(key)

        if idx2word[key] == END_TOKEN:
            # Get texts from idx for predicted sentence
            prediction = get_txt_from_idx(output_seq, idx2word)
            return prediction

        # Make current decoder output as decoder input for next time step
        dec_input = tf.expand_dims([key], 0)

    prediction = get_txt_from_idx(output_seq, idx2word)
    return prediction


class TrainerConfig:
    def __init__(self, input_vocab_size: int, target_vocab_size: int,
                 max_len_input: int, max_len_output: int,
                 embedding_dim: int, units: int, dropout: float,
                 model_name: str, batch_size: int, epochs: int, optimizer="adam",
                 loss="sparse_categorical_crossentropy",
                 metrics=["sparse_categorical_crossentropy"],
                 save_model_filepath="model_save/weights-{epoch:02d}-{loss:.4f}.hdf5",
                 monitor="val_loss", factor=0.9, verbose=1, save_best_only=True, mode="min", patience_for_lr=2,
                 patience_for_early_stop=8, min_lr=0.0001, save_weights_only=True, embedding_matrix=None,
                 restore_best_weights=True):
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.embedding_dim = embedding_dim
        self.max_len_input = max_len_input
        self.max_len_output = max_len_output
        self.units = units
        self.dropout = dropout
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.filepath = save_model_filepath
        self.monitor = monitor
        self.factor = factor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.patience_for_lr = patience_for_lr
        self.patience_for_early_stop = patience_for_early_stop
        self.model_check_point = ModelCheckpoint(filepath=self.filepath, monitor=self.monitor, verbose=self.verbose,
                                                 save_weights_only=self.save_weights_only,
                                                 save_best_only=self.save_best_only, mode=self.mode)
        self.min_lr = min_lr
        self.reduce_lr = ReduceLROnPlateau(monitor=self.monitor, factor=self.factor, verbose=self.verbose,
                                           patience=self.patience_for_lr, min_lr=self.min_lr)
        self.terminate = EarlyStopping(monitor=self.monitor, patience=self.patience_for_early_stop,
                                       verbose=self.verbose, mode=self.mode)
        self.embedding_matrix = embedding_matrix


class Trainer:
    def __init__(self, input_data: np.ndarray, teacher_data: np.ndarray, target_data: np.ndarray,
                 config: TrainerConfig):
        self.input_data = input_data
        self.teacher_data = teacher_data
        self.target_data = target_data
        self.config = config
        self.model = EncoderDecoderModel(input_vocab_size=config.input_vocab_size,
                                         target_vocab_size=config.target_vocab_size,
                                         embedding_dim=config.embedding_dim, units=config.units,
                                         embedding_matrix=config.embedding_matrix,
                                         dropout=config.dropout, name=config.model_name)
        self.model.build(input_shape=[input_data.shape, teacher_data.shape])
        pprint(self.model.summary())
        self.history = None

    def train(self):
        tf.keras.backend.clear_session()
        self.model.compile(optimizer=self.config.optimizer, loss=self.config.loss, metrics=self.config.metrics)
        check_point = self.config.model_check_point
        reduce_lr = self.config.reduce_lr
        terminate = self.config.terminate
        # callback_list = [check_point, reduce_lr, terminate]
        callback_list = [reduce_lr, terminate]
        self.history = self.model.fit([self.input_data, self.teacher_data], self.target_data,
                                      batch_size=self.config.batch_size,
                                      epochs=self.config.epochs,
                                      validation_split=0.2, callbacks=callback_list)
        self.model.save_weights("best_model_weights.h5", save_format="h5")


if __name__ == "__main__":
    X_train, _, y_train, _ = generate_word_based_train_test_dataset()
    word2idx, idx2word, vocab = generate_vocab_dict()
    embedding_matrix = generate_embedding_matrix_from_glove(vocab=vocab, word2idx=word2idx)
    input_data, teacher_data, target_data, max_len_input, max_len_output = generate_train_dataset_for_model(
        X_train=X_train, y_train=y_train, word2idx=word2idx)
    config = TrainerConfig(input_vocab_size=len(word2idx), target_vocab_size=len(word2idx), embedding_dim=300,
                           embedding_matrix=embedding_matrix,
                           units=128, max_len_input=max_len_input, max_len_output=max_len_output,
                           dropout=0.2, model_name="Encoder_Decoder", batch_size=16, epochs=50)
    trainer = Trainer(input_data, teacher_data, target_data, config)
    trainer.train()
    history = trainer.history
    plt.plot(history.history['loss'], label="Training loss")
    plt.plot(history.history['val_loss'], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("train_history.png")

    test_input_txt_seq = ["i have", "we can", "please mark", "follow these"]
    pred_steps = 4
    for i in test_input_txt_seq:
        prediction = get_predicted_txt_seq(enc_dec_model=trainer.model, word2idx=word2idx, idx2word=idx2word,
                                           output_max_length=trainer.config.max_len_output, input_seq=i)
        pprint("input seq : {}".format(i))
        if len(prediction) <= pred_steps:
            pprint("predict seq : {}".format(prediction))
        else:
            pprint("predict seq : {}".format(prediction[1:pred_steps]))
