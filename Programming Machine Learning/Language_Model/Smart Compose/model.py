import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization

from preprocess import generate_word_based_train_test_dataset, generate_embedding_matrix_from_glove, \
    generate_train_dataset_for_model, generate_vocab_dict
from pprint import pprint


class Encoder(tf.keras.layers.Layer):
    """
    Encoder: x-> embedding_layer -> bi_lstm (with dropout) -> (encoder_outputs, encoder_state_h, encoder_state_c)
    """

    def __init__(self, vocab_size: int, embedding_dim: int, units: int, embedding_matrix=None, **kwargs):
        """
        :param vocab_size: int
        :param embedding_dim: int
        :param units: int
        :param dropout: float
        :param kwargs:
        """
        super(Encoder, self).__init__()
        if embedding_matrix is None:
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True,
                                       name="Encoder_Embedding", trainable=True)
        else:
            self.vocab_size = embedding_matrix.shape[0]
            self.embedding_dim = embedding_matrix.shape[1]
            self.embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim,
                                       weights=[embedding_matrix], mask_zero=True,
                                       name="Encoder_Embedding", trainable=False)
        self.enc_units = units
        self.lstm = LSTM(units=units, return_sequences=True, return_state=True, activation='relu', name="Encoder_LSTM")

    def build(self, input_shape):
        super(Encoder, self).build(input_shape=input_shape)

    def call(self, inputs: np.ndarray) -> (tf.Tensor, tf.Tensor, tf.Tensor):
        """
        :param inputs: np.ndarray
        :return: (tf.Tensor, tf.Tensor, tf.Tensor)
        """
        x = self.embedding(inputs=inputs)
        encoder_outputs, encoder_state_h, encoder_state_c = self.lstm(x)
        return encoder_outputs, encoder_state_h, encoder_state_c


class Decoder(tf.keras.layers.Layer):
    """
    Decoder: x, (encoder_state_h, encoder_state_c) -> embedding_layer -> lstm(with dropout)
                                                   -> dense_1 (with dropout, ReLu) -> dense_2(softmax)
    """

    def __init__(self, vocab_size: int, embedding_dim: int, units: int, dropout: float,
                 embedding_matrix=None, **kwargs):
        """
        :param vocab_size: int
        :param embedding_dim: int
        :param units: int
        :param dropout: float
        :param kwargs:
        """
        super(Decoder, self).__init__()
        self.dec_units = units
        if embedding_matrix is None:
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True,
                                       name="Decoder_Embedding", trainable=True)
        else:
            self.vocab_size = embedding_matrix.shape[0]
            self.embedding_dim = embedding_matrix.shape[1]
            self.embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, mask_zero=True,
                                       weights=[embedding_matrix], name="Decoder_Embedding", trainable=False)
        self.lstm = LSTM(units=self.dec_units, return_sequences=True, return_state=True, activation='relu',
                         name="Decoder_LSTM")
        self.batch_norm = BatchNormalization()
        self.dropout_rate = dropout
        self.dropout = Dropout(rate=self.dropout_rate)
        self.dense_1 = Dense(units, activation="relu")
        self.dense_2 = Dense(vocab_size, activation="softmax")

    def build(self, input_shape):
        super(Decoder, self).build(input_shape=input_shape)

    def call(self, inputs: np.ndarray, enc_hidden: tf.Tensor, enc_cell: tf.Tensor) -> tf.Tensor:
        """
        :param inputs: np.ndarray
        :param enc_hidden: tf.Tensor
        :param enc_cell: tf.Tensor
        :return:
        """
        x = self.embedding(inputs=inputs)
        lstm_output, _, _ = self.lstm(inputs=x, initial_state=[enc_hidden, enc_cell])
        lstm_output = self.batch_norm(lstm_output)
        lstm_output = self.dense_1(lstm_output)
        lstm_output = self.dropout(lstm_output)
        decoder_outputs = self.dense_2(lstm_output)
        return decoder_outputs


class EncoderDecoderModel(tf.keras.models.Model):
    """
    EncoderDecoder: encoder -> decoder
    """

    def __init__(self, input_vocab_size: int, target_vocab_size: int, embedding_dim: int, units: int, dropout: float,
                 embedding_matrix=None, **kwargs):
        """
        :param input_vocab_size: int
        :param target_vocab_size: int
        :param embedding_dim: int
        :param units: int
        :param dropout: float
        :param kwargs:
        """
        super(EncoderDecoderModel, self).__init__()
        self.encoder = Encoder(vocab_size=input_vocab_size, embedding_dim=embedding_dim,
                               embedding_matrix=embedding_matrix, units=units, name="Encoder")
        self.decoder = Decoder(vocab_size=target_vocab_size, embedding_dim=embedding_dim,
                               embedding_matrix=embedding_matrix, units=units, name="Decoder",
                               dropout=dropout)

    def build(self, input_shape):
        super(EncoderDecoderModel, self).build(input_shape=input_shape)

    def call(self, inputs: []) -> tf.Tensor:
        """
        :param inputs: list
        :return: tf.Tensor
        """
        # Inputs will contain encoder input and decoder input
        encoder_inputs, decoder_inputs = inputs[0], inputs[1]
        # Generate output and hidden states from encoder object
        encoder_outputs, encoder_state_h, encoder_state_c = self.encoder(inputs=encoder_inputs)
        # Generate output from decoder
        # Initialize hidden states of decoder with hidden states of encoder
        outputs = self.decoder(inputs=decoder_inputs, enc_hidden=encoder_state_h, enc_cell=encoder_state_c)
        return outputs


def test_run():
    # Generate Sample Test Input Data
    X_train, X_test, y_train, y_test = generate_word_based_train_test_dataset()
    word2idx, idx2word, vocab = generate_vocab_dict()
    embedding_matrix = generate_embedding_matrix_from_glove(vocab=vocab, word2idx=word2idx)
    input_data, teacher_data, target_data, max_len_input, max_len_output = generate_train_dataset_for_model(
        X_train=X_train, y_train=y_train,
        word2idx=word2idx)
    batch_size = 32
    vocab_size = embedding_matrix.shape[0]
    embedding_dim = embedding_matrix.shape[1]
    pprint("Vocab Size : {}, Embedding Dim : {}".format(vocab_size, embedding_dim))

    # Test Encoder Behavior
    encoder = Encoder(vocab_size=vocab_size, embedding_dim=embedding_dim, embedding_matrix=embedding_matrix, units=256,
                      name="Encoder")
    example_input_batch, example_teacher_batch, example_target_batch = input_data[:batch_size], teacher_data[
                                                                                                :batch_size], \
                                                                       target_data[:batch_size]
    pprint('Shape of batch input data (batch_size,idx length): {}'.format(example_input_batch.shape))
    pprint('Shape of batch teacher data (batch_size,idx length): {}'.format(example_teacher_batch.shape))
    pprint('Shape of batch target data (batch_size,idx length): {}'.format(example_target_batch.shape))

    sample_encoder_output, sample_hidden_output, sample_cell_output = encoder(example_input_batch)
    pprint('Encoder:')
    pprint('Shape of encoder output (batch_size,units): {}'.format(sample_encoder_output.shape))
    pprint('Shape of encoder hidden state output (batch_size,units): {}'.format(sample_hidden_output.shape))
    pprint('Shape of encoder memory state output (batch_size,units): {}'.format(sample_cell_output.shape))

    # Test Decoder Behavior
    decoder = Decoder(vocab_size=vocab_size, embedding_dim=embedding_dim, embedding_matrix=embedding_matrix,
                      units=256, dropout=0.2)
    sample_output_dec = decoder(example_target_batch, sample_hidden_output, sample_cell_output)
    pprint('Decoder:')
    pprint('Shape of decoder input (batch_size,sequence_length): {}'.format(example_target_batch.shape))
    pprint('Shape of decoder output (batch_size,sequence_length,vocab_size): {}'.format(sample_output_dec.shape))

    # Test Encoder_Decoder Model Behavior
    model = EncoderDecoderModel(input_vocab_size=vocab_size, target_vocab_size=vocab_size, embedding_dim=embedding_dim,
                                units=256, dropout=0.2, embedding_matrix=embedding_matrix)
    tf.keras.backend.clear_session()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    model.fit(x=[example_input_batch, example_teacher_batch], y=example_target_batch, batch_size=batch_size, epochs=1,
              validation_split=0.2)
    pprint(model.summary())


if __name__ == "__main__":
    test_run()
