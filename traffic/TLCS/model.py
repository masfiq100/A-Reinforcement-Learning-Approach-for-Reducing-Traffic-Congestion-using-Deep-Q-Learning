import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import tensorflow as tf
import numpy as np
import sys

from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model(num_layers, width)

    def _build_model(self, num_layers, width):
        inputs = keras.Input(shape=(self._input_dim,))
        x = layers.Dense(width, activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
        for _ in range(num_layers):
            x = layers.Dense(width, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        outputs = layers.Dense(self._output_dim, activation='linear', kernel_regularizer=regularizers.l2(0.01))(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='my_model')
        optimizer = Adam(learning_rate=self._learning_rate)
        model.compile(loss=keras.losses.mean_squared_error, optimizer=optimizer)
        return model

    @property
    def batch_size(self):
        return self._batch_size

    def predict_one(self, state):
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)

    def predict_batch(self, states):
        return self._model.predict(states)

    def train_batch(self, states, q_sa, validation_states, validation_q_sa):
        early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, mode='min', verbose=1)
        
        self._model.fit(states, q_sa, batch_size=self._batch_size, epochs=50, verbose=1,
                        validation_data=(validation_states, validation_q_sa),
                        callbacks=[early_stop, reduce_lr])

    def save_model(self, path):
        model_file_path = os.path.join(path, 'trained_model.h5')
        self._model.save(model_file_path)

        model_structure_path = os.path.join(path, 'model_structure.png')
        plot_model(self._model, to_file=model_structure_path, show_shapes=True, show_layer_names=True)

        print("Model saved successfully.")


class TestModel:
    def __init__(self, input_dim, model_path):
        self._input_dim = input_dim
        self._model = self._load_my_model(model_path)

    def _load_my_model(self, model_folder_path):
        model_file_path = os.path.join(model_folder_path, 'trained_model.h5')

        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            return loaded_model
        else:
            sys.exit("Model number not found")

    def predict_one(self, state):
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)

    @property
    def input_dim(self):
        return self._input_dim
