""" Imports """

import tensorflow as tf
import tensorflow_addons as tfa
from tensorboard.plugins.hparams import api as hp
from datetime import datetime

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import data
import parameters

import model as model_generator

from tensorflow.keras.models import Sequential, Model

from keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.efficientnet import EfficientNetB0

from keras.layers.core import Flatten, Dense, Dropout
from tensorflow.keras.layers import Flatten, Layer, Reshape, Input, TimeDistributed, LSTM, GlobalAveragePooling2D, \
    InputLayer

from keras import models, optimizers

from tensorflow.keras.mixed_precision import experimental as mixed_precision

from tensorflow.keras.callbacks import ReduceLROnPlateau

import loss_wrapper

from sklearn.utils import class_weight

# tf.compat.v1.keras.layers.enable_v2_dtype_behavior()
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

import numpy as np
import tensorflow as tf
import seaborn as sn
import matplotlib.pyplot as plt
import data
from tensorflow.python.keras.models import load_model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import data2

# Disable eager execution
# tf.compat.v1.disable_eager_execution()

# nvidia-smi -l 1

# rm -rf ./logs/


""" Hparms """
# HP_LSTM = hp.HParam('LSTM_para', hp.Discrete([128, 256]))
HP_clip_size = hp.HParam('batch', hp.Discrete([8, 16, 32]))
# HP_batch_size = hp.HParam('batch', hp.Discrete([32, 64]))
HP_LR = hp.HParam('LR', hp.Discrete([1e-02, 1e-03, 1e-05]))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs_2INPUTS_FocLoss_Adam_convLSTM_3D_DA/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_clip_size, HP_LR],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
    )

logdir = "logs_2INPUTS_FocLoss_Adam_convLSTM_3D_DA/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

""" Data """
# train_images, train_labels, test_images, test_labels, val_images, val_labels = data.my_data()
train_images_horiz, train_images_verti, train_labels, test_images_horiz, test_images_verti, test_labels, val_images_horiz, val_images_verti, val_labels = data.my_data_2input()

one_hot_train_labels = tf.one_hot(train_labels, depth=5)
one_hot_test_labels = tf.one_hot(test_labels, depth=5)
one_hot_val_labels = tf.one_hot(val_labels, depth=5)

""" Function in which the model is trained """


def train_test_model(hparams, training_generator, test_generator, val_generator, model):
    clip_size = hparams[HP_clip_size]
    # steps = int((train_images.shape[1]/clip_size))$
    steps = int((train_images_horiz.shape[1] / clip_size))
    # steps = 5

    callbacks = [
        tf.keras.callbacks.TensorBoard(logdir),  # log metrics
        hp.KerasCallback(logdir, hparams),  # log hparams
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=17),
        # tf.keras.callbacks.LearningRateScheduler(scheduler)
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
    ]

    val_steps = int((val_images_horiz.shape[1] / clip_size))
    history = model.fit(training_generator, verbose=2, epochs=200, callbacks=callbacks, steps_per_epoch=steps,
                        validation_data=val_generator, validation_steps=val_steps)

    test_steps = int((test_images_horiz.shape[1] / clip_size))
    _, accuracy = model.evaluate(test_generator, steps=test_steps)

    return accuracy, history


""" Intermediate function to initialise tensorboard logs """


def run(run_dir, hparams, training_generator, test_generator, val_generator, model):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy, history = train_test_model(hparams, training_generator, test_generator, val_generator, model)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
        return accuracy, history


session_num = 0

dossier_global = "grid_search_2INPUTS_FocLoss_Adam_convLSTM_3D_DA"
os.makedirs(dossier_global, exist_ok=True)

""" Loops : training amoung those parameters """
# for LSTM_para in HP_LSTM.domain.values:
# for batch_sizes in HP_batch_size.domain.values:
for clip_sizes in HP_clip_size.domain.values:
    for LR in HP_LR.domain.values:

        hparams = {
            # HP_batch_size: batch_sizes,
            HP_clip_size: clip_sizes,
            # HP_LSTM: LSTM_para,
            HP_LR: LR,
        }

        clip_size = clip_sizes
        """ generator for 1 input :
        training_generator = data2.my_data_generator(train_images, one_hot_train_labels, clip_size)
        test_generator = data2.my_data_generator(test_images, one_hot_test_labels, clip_size)
        val_generator = data2.my_data_generator(val_images, one_hot_val_labels, clip_size)
        """

        """ Initializing the generators """
        training_generator = data2.my_data_generator_2input(train_images_horiz, train_images_verti,
                                                            one_hot_train_labels, clip_size)
        test_generator = data2.my_data_generator_2input(test_images_horiz, test_images_verti, one_hot_test_labels,
                                                        clip_size)
        val_generator = data2.my_data_generator_2input(val_images_horiz, val_images_verti, one_hot_val_labels,
                                                       clip_size)

        """ Initializing the model """
        # model = model_generator.TD_2DCNN_RNN_multi_input(nb_towers=2, nb_classes=5, list_input_shapes=[[None, 100, 100, 1], [None, 100, 100, 1]], batch_size=1, bidirectional=False, stateful=True, return_sequences = True, RNN_layer="ConvLSTM")
        model = model_generator.ThreeDCNN_RNN_multi_input(nb_towers=2, nb_classes=5,
                                                          list_input_shapes=[[None, 100, 100, 1], [None, 100, 100, 1]],
                                                          batch_size=1, bidirectional=False, stateful=True,
                                                          return_sequences=True, RNN_layer="ConvLSTM")
        # model =  model_generator.ThreeDCNN_RNN_multi_input_simplifie(nb_towers=2, nb_classes=5, list_input_shapes=[[None, 100, 100, 1], [None, 100, 100, 1]], batch_size=1, bidirectional=False, stateful=True, return_sequences = True, RNN_layer="ConvLSTM")

        """ Parameters """
        opt = tf.keras.optimizers.Adam(learning_rate=LR)
        # opt = tf.keras.optimizers.SGD(learning_rate=LR, momentum = 0.9)

        originalLossFunc = tfa.losses.SigmoidFocalCrossEntropy()
        # originalLossFunc = tf.keras.losses.CategoricalCrossentropy()

        classes = np.arange(5)
        weightsList = class_weight.compute_class_weight('balanced', classes, np.squeeze(train_labels, axis=0))

        perso_loss = loss_wrapper.weightedLoss(originalLossFunc, weightsList)

        model.compile(
            optimizer=opt,
            loss=perso_loss,
            metrics=['accuracy'],
        )

        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        accuracy, history = run('logs_2INPUTS_FocLoss_Adam_convLSTM_3D_DA/hparam_tuning/' + run_name, hparams,
                                training_generator, test_generator, val_generator, model)
        session_num += 1

        """ Plots : confusion matrix. Loop to plot for both test and training. """
        nom_dossier = dossier_global + "/clip_size : " + str(clip_sizes) + ",learning_rate = " + str(
            LR)  # "/batch_size : " + str(batch_sizes)
        os.makedirs(nom_dossier, exist_ok=True)

        liste_a_tester = [test_generator, training_generator]
        compteur = 0

        steps = int((test_images_horiz.shape[1] / clip_size))

        for liste in liste_a_tester:
            labels = []
            predictions = []

            for step in range(steps):
                current_data, current_labels = next(test_generator)
                prediction = model.predict(current_data)

                labels.append(current_labels)
                predictions.append(prediction)

            conc_labels = np.concatenate(labels, axis=1)
            conc_labels = np.squeeze(conc_labels, axis=0)
            conc_predictions = np.concatenate(predictions, axis=1)
            conc_predictions = np.squeeze(conc_predictions, axis=0)

            np_labels_conc = np.argmax(conc_labels, -1)
            predic = np.argmax(conc_predictions, -1)

            plt.figure()
            conf = tf.math.confusion_matrix(np_labels_conc, predic)
            conf = np.array(conf, dtype=np.float32)

            # Normalier entre 0 et 1 pour chaque geste
            long_conf = len(conf)
            for i in range(long_conf):
                nb_frames = sum(conf[i])
                valeur_max_conf_geste = 1
                for j in range(long_conf):
                    valeur_courante = conf[i][j]
                    if valeur_courante > valeur_max_conf_geste:
                        valeur_max_conf_geste = valeur_courante

                conf[i] = conf[i] / nb_frames

            sn.set(font_scale=1)

            sn.heatmap(conf, annot=True, annot_kws={"size": 10})
            if compteur == 0:
                nom = 'test'
            else:
                nom = 'train'
            plt.savefig(nom_dossier + '/confusion_matrix_' + nom + '.png')
            compteur = 1
            steps = int((train_images_horiz.shape[1] / clip_size))

        """ Loss and accuracy plots """
        # "Train_data"
        plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['loss'])
        plt.title('Train data : (test_acc :' + str(accuracy) + ')')
        plt.xlabel('epoch')
        plt.legend(['Accuracy', 'Loss'], loc='upper left')
        plt.savefig(nom_dossier + '/training_acc_loss.png')

        # "Validation data"
        plt.figure()
        plt.plot(history.history['val_accuracy'])
        plt.plot(history.history['val_loss'])
        plt.title('Validation data')
        plt.xlabel('epoch')
        plt.legend(['Accuracy', 'Loss'], loc='upper left')
        plt.savefig(nom_dossier + '/validation_acc_loss.png')

# tensorboard --logdir logs/hparam_tuning