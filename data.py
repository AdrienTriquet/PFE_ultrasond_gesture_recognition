""" Import """

import numpy as np
import imgaug.augmenters as iaa

# import labels from doc excel
import pandas as pd
import os

from goto import with_goto

import cv2

""" Return the concatenated lists of labels, 1 list per information out of a csv file """


def list_labels_out_of_csv(path):
    # création des listes des différents labels
    label_num = []
    label_nom = []
    t_in = []
    t_out = []

    # On va évoluer à travers les différents fichiers de labels (1 par paire de vidéo)

    listing = os.listdir(path)
    for file in listing:
        # Vérification qu'on ouvre bien le bon fichier
        if file.startswith('ID'):
            # Ouverture
            label_path = path + '/' + file
            # header=None sinon première ligne considéré comme nom et pas prise en compte
            document = pd.read_csv(label_path, header=None)
            labels = document.to_numpy()

            # on crée une liste par fichier
            label_num.append([])
            label_nom.append([])
            t_in.append([])
            t_out.append([])

            # pour chaque ligne du fichier, donc de l'aurray 'labels'
            for i in range(len(labels)):
                # mise sous le bon format des données
                ligne = labels[i][0].split(';')

                # ajout aux listes de labels à la liste actuelle
                label_num[-1].append(int(ligne[0]))
                label_nom[-1].append(ligne[1])
                t_in[-1].append(int(ligne[2]))
                t_out[-1].append(int(ligne[3]))

    return label_num, label_nom, t_in, t_out


""" Take out all the frames of a video and put them in a list """


def list_frames_out_of_videos(path):
    # Creation liste générale avec toutes les frames de toutes les videos
    list_videos = []

    # On parcourt tous les fichiers
    listing = os.listdir(path)
    for file in listing:
        if file.startswith('ID'):
            video_path = path + '/' + file

            # Opens the Video file
            cap = cv2.VideoCapture(video_path)

            # Put the frames of this video in a list
            list_frames = []
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret == False:
                    break
                # Passage sous niveaux de gris
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                list_frames.append(gray)

            # On obtient ainsi une liste avec 1 liste par video
            list_videos.append(list_frames)

    cap.release()
    cv2.destroyAllWindows()

    return list_videos


""" Creation des listes de travail : des frames t_in à t_out on est dans un même 'label_num' """


@with_goto
def list_input_network_creation(path_videos, path_labels):
    # Creation des listes de travail
    train_images = []
    train_labels = []

    compteur_gestes = 0

    list_videos = list_frames_out_of_videos(path_videos)
    nb_videos = len(list_videos)

    labels = list_labels_out_of_csv(path_labels)
    label_num = labels[0]
    t_in = labels[2]
    t_out = labels[3]

    # On évolue une vidéo après l'autre
    for video in range(len(list_videos)):
        # Pour savoir quel t_in et t_out prendre, on compte les mouvements
        nb_mouvement = len(t_in[video])
        mouvement_compteur = 0

        # Dans chaque vidéo, on regarde frame par frame
        nb_frame = len(list_videos[video])
        for frame in range(nb_frame):

            # Quand on attend le nombre de mouvement : on peut s'arreter
            if mouvement_compteur == nb_mouvement:
                goto.start

            """ Si on est entre 2 marqueurs t_in et t_out alors notre vidéo est un 
            enregistrement d'une séquence de mouvement, alors on met la frame et 
            le numéro du mouvement dans les listes de travail """
            # < stricte car le mouvement ne commence et ne finit jamais sur cette frame
            if frame > t_in[video][mouvement_compteur] and frame < t_out[video][mouvement_compteur]:
                train_images.append(list_videos[video][frame])
                train_labels.append(label_num[video][mouvement_compteur])

            # Si on est à t_out, le mouvement est fini
            elif frame == t_out[video][mouvement_compteur]:
                train_images.append(list_videos[video][frame])
                train_labels.append(label_num[video][mouvement_compteur])

                # On passe au mouvement suivant pour l'indice de t_in et t_out
                mouvement_compteur += 1

            # pour toutes les images hors 'recording', on met en 'random' cad 4
            else:
                label.start
                train_images.append(list_videos[video][frame])
                train_labels.append(4)

    return train_images, train_labels


""" Generation and pre processing of all lists needed """


def list_input_network():
    """train images"""
    path_train_labels = "/space/homes/atriquet/PFE/dev/data/train_data/Labels"
    path_train_videos_horiz = "/space/homes/atriquet/PFE/dev/data/train_data/Horizontal"
    path_train_videos_verti = "/space/homes/atriquet/PFE/dev/data/train_data/Vertical"
    train_images_horiz, train_labels = list_input_network_creation(path_train_videos_horiz, path_train_labels)
    train_images_verti, train_labels = list_input_network_creation(path_train_videos_verti, path_train_labels)

    """test images"""
    path_test_labels = "/space/homes/atriquet/PFE/dev/data/test_data/Labels"
    path_test_videos_horiz = "/space/homes/atriquet/PFE/dev/data/test_data/Horizontal"
    path_test_videos_verti = "/space/homes/atriquet/PFE/dev/data/test_data/Vertical"
    test_images_horiz, test_labels = list_input_network_creation(path_test_videos_horiz, path_test_labels)
    test_images_verti, test_labels = list_input_network_creation(path_test_videos_verti, path_test_labels)

    """val images"""
    path_val_labels = "/space/homes/atriquet/PFE/dev/data/val_data/Labels"
    path_val_videos_horiz = "/space/homes/atriquet/PFE/dev/data/val_data/Horizontal"
    path_val_videos_verti = "/space/homes/atriquet/PFE/dev/data/val_data/Vertical"
    val_images_horiz, val_labels = list_input_network_creation(path_val_videos_horiz, path_val_labels)
    val_images_verti, val_labels = list_input_network_creation(path_val_videos_verti, path_val_labels)

    # passage sous la forme d'array
    train_images_horiz = np.array(train_images_horiz)
    train_images_verti = np.array(train_images_verti)
    train_labels = np.array(train_labels)

    test_images_horiz = np.array(test_images_horiz)
    test_images_verti = np.array(test_images_verti)
    test_labels = np.array(test_labels)

    val_images_horiz = np.array(val_images_horiz)
    val_images_verti = np.array(val_images_verti)
    val_labels = np.array(val_labels)

    # pre-processing : passage de 0-255 à 0-1 pour les valeurs de niveaux de gris, normalisation
    train_images_horiz = train_images_horiz / 255.0
    train_images_verti = train_images_verti / 255.0

    test_images_horiz = test_images_horiz / 255.0
    test_images_verti = test_images_verti / 255.0

    val_images_horiz = val_images_horiz / 255.0
    val_images_verti = val_images_verti / 255.0

    """ manque le nombre de channel, donc 1 car nuance de gris """
    train_images_horiz = np.expand_dims(train_images_horiz, -1)
    train_images_verti = np.expand_dims(train_images_verti, -1)

    test_images_horiz = np.expand_dims(test_images_horiz, -1)
    test_images_verti = np.expand_dims(test_images_verti, -1)

    val_images_horiz = np.expand_dims(val_images_horiz, -1)
    val_images_verti = np.expand_dims(val_images_verti, -1)

    """ Adding a dimension in first place to be concatened (clip size) """
    train_images_horiz = np.expand_dims(train_images_horiz, 0)
    train_images_verti = np.expand_dims(train_images_verti, 0)
    train_labels = np.expand_dims(train_labels, 0)

    test_images_horiz = np.expand_dims(test_images_horiz, 0)
    test_images_verti = np.expand_dims(test_images_verti, 0)
    test_labels = np.expand_dims(test_labels, 0)

    val_images_horiz = np.expand_dims(val_images_horiz, 0)
    val_images_verti = np.expand_dims(val_images_verti, 0)
    val_labels = np.expand_dims(val_labels, 0)

    return train_images_horiz, train_images_verti, train_labels, test_images_horiz, test_images_verti, test_labels, val_images_horiz, val_images_verti, val_labels


""" Data augmentatio basis : rotation, translation, cropping """
def augmentation():
    train_images_horiz, train_images_verti, train_labels, test_images_horiz, test_images_verti, test_labels, val_images_horiz, val_images_verti, val_labels = list_input_network()

    seq1 = iaa.Sequential([
        iaa.Fliplr(1)
    ])

    seq2 = iaa.Sequential([
        iaa.Flipud(1)
    ])

    seq3 = iaa.Sequential([
        iaa.Rotate((-45, 45))
    ])

    seq4 = iaa.Sequential([
        iaa.TranslateX(px=(-20, 20))
    ])

    seq5 = iaa.Sequential([
        iaa.CropToFixedSize(width=30, height=30)
    ])

    new_train_images = train_images_horiz
    new_train_labels = train_labels

    """ different sequences of data augmentation applied """
    images_aug = seq1(images=train_images_horiz)
    new_train_images = np.concatenate((new_train_images, images_aug))
    new_train_labels = np.concatenate((new_train_labels, train_labels))

    images_aug = seq2(images=train_images_horiz)
    new_train_images_horiz = np.concatenate((new_train_images, images_aug))
    new_train_labels = np.concatenate((new_train_labels, train_labels))

    images_aug = seq3(images=train_images_horiz)
    new_train_images = np.concatenate((new_train_images, images_aug))
    new_train_labels = np.concatenate((new_train_labels, train_labels))

    images_aug = seq4(images=train_images_horiz)
    new_train_images_horiz = np.concatenate((new_train_images, images_aug))
    new_train_labels = np.concatenate((new_train_labels, train_labels))

    #new_train_images = train_images_verti

    """ same on vertical images """
    """
    images_aug = seq1(images=train_images_verti)
    new_train_images = np.concatenate((new_train_images, images_aug))

    images_aug = seq2(images=train_images_verti)
    new_train_images_verti = np.concatenate((new_train_images, images_aug))

    images_aug = seq3(images=train_images_verti)
    new_train_images = np.concatenate((new_train_images, images_aug))

    images_aug = seq4(images=train_images_verti)
    new_train_images_verti = np.concatenate((new_train_images, images_aug))
    """

    # images_aug = seq5(images=train_images)
    # new_train_images = np.concatenate((new_train_images, images_aug))
    # new_train_labels = np.concatenate((new_train_labels, train_labels))

    #return new_train_images_horiz, new_train_images_verti, new_train_labels
    return new_train_images_horiz, new_train_labels

""" Function to call in others files for 1 input, return data with or without augmentation """
def my_data():
    train_images_horiz, train_images_verti, train_labels, test_images_horiz, test_images_verti, test_labels, val_images_horiz, val_images_verti, val_labels = list_input_network()

    """ Avec data augmentation """
    #train_images_horiz, train_images_verti, train_labels = augmentation()

    # return train_images_final, train_labels, test_images_final, test_labels
    return train_images_horiz, train_labels, test_images_horiz, test_labels, val_images_horiz, val_labels

""" Function to call in others files for 2 inputs, return data with or without augmentation """
def my_data_2input():
    train_images_horiz, train_images_verti, train_labels, test_images_horiz, test_images_verti, test_labels, val_images_horiz, val_images_verti, val_labels = list_input_network()

    """ Avec data augmentation """
    #train_images_horiz, train_images_verti, train_labels = augmentation()

    # return train_images_final, train_labels, test_images_final, test_labels
    return train_images_horiz, train_images_verti, train_labels, test_images_horiz, test_images_verti, test_labels, val_images_horiz, val_images_verti, val_labels
