import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow_addons as tfa
import keras

from tensorflow.keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50

from keras import models, optimizers

import tensorflow as tf

from dl_building import ThreeD_ConvLayers_block, ConvLayers_block, LSTM_block, DenseLayers_block
from dl_building import Swish, attention_layer, attention3D_layer, BahdanauAttention

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, Dropout, Flatten, Input, \
    Activation, BatchNormalization, Softmax, LeakyReLU, concatenate, InputLayer
from tensorflow.keras.layers import TimeDistributed, Bidirectional, LSTM, ConvLSTM2D, GlobalAveragePooling1D, \
    GlobalAveragePooling2D, GlobalAveragePooling3D, Reshape, Lambda, GRU
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.activations import relu


def model_grid_2(hparams):
    vggmodel = VGG16(weights=None, include_top=False, classes=5, pooling="avg", input_shape=(100, 100, 1))
    input_layer = Input(batch_shape=(16, 100, 100, 1))
    h2 = TimeDistributed(vggmodel)(input_layer)
    encoded_sequence = LSTM(128)(h2)
    hidden_layer = Dense(512, activation="relu")(encoded_sequence)
    outputs = Dense(5, activation="softmax")(hidden_layer)
    model = Model([input_layer], outputs)

    return model


def model_grid(hparams):
    vggmodel = VGG16(weights=None, include_top=False, classes=5, pooling="avg", input_shape=(100, 100, 1))
    model = Sequential()
    model.add(TimeDistributed(vggmodel, input_shape=(1, 100, 100, 1)))
    model.add(LSTM(10))
    model.add(Dense(5, activation="softmax"))

    return model


def generate_model():
    baseModel = VGG16(weights=None, include_top=False, classes=5, pooling="avg", input_shape=(100, 100, 1))
    # baseModel = ResNet50(weights= None, include_top=False, classes=5, pooling="avg", input_shape=(100, 100, 1))
    # baseModel = EfficientNetB0(weights= None, include_top=False, classes=5, pooling="avg", input_shape=(100, 100, 1))

    model = models.Sequential()

    # model.add(InputLayer(batch_size=hparams[HP_batch_size]))
    model.add(InputLayer(batch_input_shape=(16, 100, 100, 1)))

    model.add(TimeDistributed(baseModel))

    # model.add(InputLayer(batch_size=hparams[HP_batch_size]))
    # model.add(InputLayer(batch_input_shape=hparams[HP_batch_size]))

    # model.add(Dense(2048, activation='relu'))

    # model.add(Dropout(0.5))

    # model.add(Dense(2048, activation='relu'))

    # model.add(Reshape(target_shape = (2048,1)))

    # model.add(LSTM(hparams[HP_LSTM], return_sequences=True))

    # model.add(Dropout(0.5))  batch_input_shape = (, 2048)

    model.add(LSTM(128, stateful=True))

    model.add(Dense(2048, activation='relu'))

    model.add(Dense(5, activation='softmax'))

    tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer="Adam",
        loss=tfa.losses.SigmoidFocalCrossEntropy(),
        # loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model