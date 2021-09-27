import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from model import EncoderDecoderModel


class TrainerConfig:
    def __init__(self, input_vocab_size: int, output_vocab_size: int, embedding_dim: int, units: int, dropout: float,
                 model_name: str, batch_size: int, epochs: int, optimizer="adam",
                 loss="sparse_categorical_crossentropy",
                 metrics=["sparse_categorical_crossentropy"],
                 save_model_filepath="model_save/weights-{epoch:02d}-{loss:.4f}.hdf5",
                 monitor="val_loss", factor=0.9, verbose=1, save_best_only=True, mode="min", patience_for_lr=2,
                 patience_for_early_stop=8, min_lr=0.0001, restore_best_weights=True):
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
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
        self.reduce_lr = ReduceLROnPlateau(monitor=self.monitor, factor=self.factor, verbose=self.verbose, patience=self.patience_for_lr, min_lr=self.min_lr)
        self.terminate = EarlyStopping(monitor=self.monitor, patience=self.patience_for_early_stop, verbose=self.verbose, mode=self.mode, )


class Trainer:
    def __init__(self, model: EncoderDecoderModel, train_dataset: np.ndarray, test_dataset: np.ndarray,
                 config: TrainerConfig):
        pass
