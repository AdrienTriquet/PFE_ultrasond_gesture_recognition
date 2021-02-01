""" create a generator for 1 input data """
def my_data_generator_1input(images, labels, clip_size):
    while True:
        # load data & labels (one hot encoded)
        data = images
        labels = labels

        nb_frames = data.shape[1]
        for clip_index in range(int(nb_frames / clip_size)):
            # current clip & labels
            current_clip = data[:, clip_size * clip_index:clip_size * (clip_index + 1), :, :, :]
            current_labels = labels[:, clip_size * clip_index:clip_size * (clip_index + 1), :]

            yield current_clip, current_labels
            del current_clip
            del current_labels


""" create a generator for 2 inputs data """
def my_data_generator_2input(images_horiz, images_verti, labels, clip_size):
    while True:  # it needs to loop infinetly
        # print("hello while, you are in")
        # load your data & labels
        data_horiz = images_horiz
        data_verti = images_verti
        labels = labels  # your labels must be one hot encoded not categorical

        # Shuffle if you need to
        # Do your data augmentation if you need to

        nb_frames = data_horiz.shape[1]
        # print(str(nb_frames))
        # data = np.expand_dims(data, axis=0) # fix the syntax if it is wrong
        # labels = np.expand_dims(data, axis=0) # fix the syntax if it is wrong

        for clip_index in range(int(nb_frames / clip_size)):
            # current clip & labels
            current_clip_horiz = data_horiz[:, clip_size * clip_index:clip_size * (clip_index + 1), :, :, :]
            current_clip_verti = data_verti[:, clip_size * clip_index:clip_size * (clip_index + 1), :, :, :]
            current_labels = labels[:, clip_size * clip_index:clip_size * (clip_index + 1), :]

            yield [current_clip_horiz, current_clip_verti], current_labels
            del current_clip_horiz
            del current_clip_verti
            del current_labels
