import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Concatenate
import matplotlib.pyplot as plt
from pprint import pprint


class Encoder(tf.keras.layers.Layer):
    """
    Encoder: x-> embedding_layer -> bi_lstm (with dropout) -> (encoder_outputs, encoder_state_h, encoder_state_c)
    """

    def __init__(self, vocab_size: int, embedding_dim: int, units: int, **kwargs):
        """
        :param vocab_size: int
        :param embedding_dim: int
        :param units: int
        :param dropout: float
        :param kwargs:
        """
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.enc_units = units
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True,
                                   name="Encoder_Embedding", trainable=True)
        self.bi_lstm = Bidirectional(
            LSTM(units=units, return_sequences=True, return_state=True, activation='relu', name="Encoder_BiLSTM"))

    def call(self, inputs: np.ndarray) -> (tf.Tensor, tf.Tensor, tf.Tensor):
        """
        :param inputs: np.ndarray
        :return: (tf.Tensor, tf.Tensor, tf.Tensor)
        """
        x = self.embedding(inputs=inputs)
        encoder_outputs, f_state_h, f_state_c, b_state_h, b_state_c = self.bi_lstm(x)
        encoder_state_h = Concatenate()([f_state_h, b_state_c])
        encoder_state_c = Concatenate()([f_state_c, b_state_c])
        return encoder_outputs, encoder_state_h, encoder_state_c


class Decoder(tf.keras.layers.Layer):
    """
    Decoder: x, (encoder_state_h, encoder_state_c) -> embedding_layer -> lstm(with dropout)
                                                   -> dense_1 (with dropout, ReLu) -> dense_2(softmax)
    """

    def __init__(self, vocab_size: int, embedding_dim: int, units: int,
                 dropout: float, **kwargs):
        """
        :param vocab_size: int
        :param embedding_dim: int
        :param units: int
        :param dropout: float
        :param kwargs:
        """
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dec_units = units * 2
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name="Decoder_Embedding")
        self.lstm = LSTM(units=self.dec_units, return_sequences=True, return_state=True, activation='relu',
                         name="Decoder_LSTM")
        self.dropout_rate = dropout
        self.dropout = Dropout(rate=self.dropout_rate)
        self.dense_1 = Dense(units, activation="relu")
        self.dense_2 = Dense(vocab_size, activation="softmax")

    def call(self, inputs: np.ndarray, enc_hidden: tf.Tensor, enc_cell: tf.Tensor) -> tf.Tensor:
        """
        :param inputs: np.ndarray
        :param enc_hidden: tf.Tensor
        :param enc_cell: tf.Tensor
        :return:
        """
        x = self.embedding(inputs=inputs)
        lstm_output, _, _ = self.lstm(inputs=x, initial_state=[enc_hidden, enc_cell])
        lstm_output = self.dense_1(lstm_output)
        lstm_output = self.dropout(lstm_output)
        decoder_outputs = self.dense_2(lstm_output)
        return decoder_outputs


class EncoderDecoderModel(tf.keras.models.Model):
    """
    EncoderDecoder: encoder -> decoder
    """

    def __init__(self, input_vocab_size: int, output_vocab_size: int, embedding_dim: int, units: int, dropout: float,
                 **kwargs):
        """
        :param input_vocab_size: int
        :param output_vocab_size: int
        :param embedding_dim: int
        :param units: int
        :param dropout: float
        :param kwargs:
        """
        super(EncoderDecoderModel, self).__init__()
        self.encoder = Encoder(vocab_size=input_vocab_size, embedding_dim=embedding_dim, units=units, name="Encoder")
        self.decoder = Decoder(vocab_size=output_vocab_size, embedding_dim=embedding_dim, units=units, name="Decoder",
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


def test_run(input_vocab_size=10000, output_vocab_size=10000, embedding_dim=300, units=64, dropout=0.2,
             name="Encoder_Decoder_Model"):
    model = EncoderDecoderModel(input_vocab_size=input_vocab_size, output_vocab_size=output_vocab_size,
                                embedding_dim=embedding_dim, units=units, dropout=dropout, name=name)
    input_seq_len = 50
    output_seq_len = 50
    batch_size = 64
    sample_size = batch_size * 20
    encoder_inputs = np.random.randint(0, input_vocab_size, size=(sample_size, input_seq_len))
    decoder_inputs = np.random.randint(0, output_vocab_size, size=(sample_size, output_seq_len))
    target = np.random.randint(0, output_vocab_size, size=(sample_size, output_seq_len))
    model.build(input_shape=[encoder_inputs.shape, decoder_inputs.shape])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_crossentropy"])
    pprint(model.summary())
    history = model.fit([encoder_inputs, decoder_inputs], target, batch_size=batch_size, epochs=10,
                        validation_split=0.2)
    plt.plot(history.history['loss'], label="Training loss")
    plt.plot(history.history['val_loss'], label="Validation loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_run()
