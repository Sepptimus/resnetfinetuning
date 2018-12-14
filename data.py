
from os import listdir
# from os.path import join
import numpy as np
import config
from keras_vggface.utils import preprocess_input
import cv2
# from abc import ABCMeta,abstractmethod
from os.path import join, exists, isdir


class DataReader:
    def __init__(self, dir_images):
        self.root = dir_images
        self.list_classes = listdir(self.root)

        self.list_classes_idx = range(len(self.list_classes))
        # Set weights in case of imbalanced number of observations per identities
        self.weights = [len(listdir(join(self.root, c))) for c in self.list_classes]
        self.weights = np.array(self.weights)
        self.weights = self.weights / np.sum(self.weights)

    def MakeTriplet(self):
        # positive and anchor classes are selected from folders where have more than two pictures
        idx_class_pos = np.random.choice(self.list_classes_idx, 1, p=self.weights)[0]
        name_pos = self.list_classes[idx_class_pos]
        dir_pos = join(self.root, name_pos)
        [img_anchor_name, img_pos_name] = np.random.choice(listdir(dir_pos), 2, replace=False)

        # negative classes are selected from all folders
        while True:
            idx_class_neg = np.random.choice(self.list_classes_idx, 1, p=self.weights)[0]
            if idx_class_neg != idx_class_pos:
                break
        name_neg = self.list_classes[idx_class_neg]
        dir_neg = join(self.root, name_neg)
        img_neg_name = np.random.choice(listdir(dir_neg), 1)[0]

        path_anchor = join(dir_pos, img_anchor_name)
        path_pos = join(dir_pos, img_pos_name)
        path_neg = join(dir_neg, img_neg_name)
        return path_anchor, path_pos, path_neg


def _ImagePreprocess(path):
    im = cv2.imread(path)
    im = cv2.resize(im, (config.image_size, config.image_size))
    # im = np.expand_dims(im, axis=0)
    im = preprocess_input(im.astype(np.float64), version=2)
    return im


def _Flip(im_array):
    if np.random.uniform(0, 1) > 0.7:
        im_array = np.fliplr(im_array)
    return im_array


def TripletGenerator(reader, label=None):
    while True:
        list_pos = []
        list_anchor = []
        list_neg = []

        for _ in range(config.batch_size):
            path_anchor, path_pos, path_neg = reader.MakeTriplet()
            img_anchor = _Flip(_ImagePreprocess(path_anchor))
            img_pos = _Flip(_ImagePreprocess(path_pos))
            img_neg = _Flip(_ImagePreprocess(path_neg))
            list_pos.append(img_pos)
            list_anchor.append(img_anchor)
            list_neg.append(img_neg)
            # print(path_anchor)
            # print(path_pos)
            # print(path_neg)

        A = np.array(list_anchor)
        P = np.array(list_pos)
        N = np.array(list_neg)

        yield ({'anchor_input': A, 'positive_input': P, 'negative_input': N}, label)


if __name__ == '__main__':
    k = DataReader(dir_images=config.test_data)
    s = list(TripletGenerator(k))
    print(s)