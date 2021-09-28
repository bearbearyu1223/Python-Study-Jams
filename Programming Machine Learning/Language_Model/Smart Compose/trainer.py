import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from model import EncoderDecoderModel
from preprocess import generate_train_test_dataset, generate_train_dataset_for_model, generate_vocab_dict
from pprint import pprint
import matplotlib.pyplot as plt


class TrainerConfig:
    def __init__(self, input_vocab_size: int, target_vocab_size: int, embedding_dim: int, units: int, dropout: float,
                 model_name: str, batch_size: int, epochs: int, optimizer="adam",
                 loss="sparse_categorical_crossentropy",
                 metrics=["sparse_categorical_crossentropy"],
                 save_model_filepath="model_save/weights-{epoch:02d}-{loss:.4f}.hdf5",
                 monitor="val_loss", factor=0.9, verbose=1, save_best_only=True, mode="min", patience_for_lr=2,
                 patience_for_early_stop=8, min_lr=0.0001, restore_best_weights=True):
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.embedding_dim = embedding_dim
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
        self.mode = mode
        self.patience_for_lr = patience_for_lr
        self.patience_for_early_stop = patience_for_early_stop
        self.model_check_point = ModelCheckpoint(filepath=self.filepath, monitor=self.monitor, verbose=self.verbose,
                                                 save_best_only=self.save_best_only, mode=self.mode)
        self.min_lr = min_lr
        self.reduce_lr = ReduceLROnPlateau(monitor=self.monitor, factor=self.factor, verbose=self.verbose,
                                           patience=self.patience_for_lr, min_lr=self.min_lr)
        self.terminate = EarlyStopping(monitor=self.monitor, patience=self.patience_for_early_stop,
                                       verbose=self.verbose, mode=self.mode, )


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
        callback_list = [check_point, reduce_lr, terminate]
        self.history = self.model.fit([self.input_data, self.teacher_data], self.target_data,
                                      batch_size=self.config.batch_size,
                                      epochs=self.config.epochs,
                                      validation_split=0.2, callbacks=callback_list)
        self.model.save_weights("best_model_weights.h5")


if __name__ == "__main__":
    X_train, _, y_train, _ = generate_train_test_dataset()
    word2idx, idx2word, vocab = generate_vocab_dict()
    input_data, teacher_data, target_data = generate_train_dataset_for_model(X_train=X_train, y_train=y_train,
                                                                             word2idx=word2idx)
    config = TrainerConfig(input_vocab_size=len(word2idx), target_vocab_size=len(word2idx), embedding_dim=300,
                           units=128,
                           dropout=0.2, model_name="Encoder_Decoder", batch_size=64, epochs=20)
    trainer = Trainer(input_data, teacher_data, target_data, config)
    trainer.train()
    history = trainer.history
    plt.plot(history.history['loss'], label="Training loss")
    plt.plot(history.history['val_loss'], label="Validation loss")
    plt.legend()
    plt.savefig("train_history.png")
