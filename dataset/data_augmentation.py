import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2


class DataAugmentation:
    def __init__(self, dataset, labels) -> None:

        self.dataset = dataset
        self.labels = labels

    def augmentation(self, flip=False, noise_levels=[], rotation=[]):

        dataset = self.dataset
        labels = self.labels

        # flip version
        if flip:
            d_flip = self.__flip()
            dataset = np.concatenate((dataset, d_flip), axis=0)
            # duplicate labels
            labels = np.concatenate((labels, self.labels), axis=0)

        for noise in noise_levels:
            d_noise = self.__add_noise(noise)
            dataset = np.concatenate((dataset, d_noise), axis=0)
            # duplicate labels
            labels = np.concatenate((labels, self.labels), axis=0)

        for angle in rotation:
            d_rot = self.__rotation(angle)
            dataset = np.concatenate((dataset, d_rot), axis=0)
            # duplicate labels
            labels = np.concatenate((labels, self.labels), axis=0)

        return dataset, labels

    def __add_noise(self, sigma=4):
        z = np.random.random(self.dataset.shape) * sigma

        return self.dataset + z

    def __flip(self):
        M = np.max(self.dataset, axis=(0, 1, 2))
        f = np.array([-1, 1])
        return self.dataset * f + np.array([M[0], 0])

    def __rotation(self, angle, img_shape=(200, 300)):
        center = np.array(img_shape) / 2

        # convert degrees to radians
        angle = 3.14 * angle / 180

        # compute centered dataset
        ctr_dataset = self.dataset - center

        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])

        return ctr_dataset @ R + center


def main():

    h5f = h5py.File('./dataset/data_train_labels.h5', 'r')
    train_labels = h5f['data_train_labels'][:]
    h5f.close()
    h5f = h5py.File('./dataset/data_train_landmarks.h5', 'r')
    train_landmarks = h5f['data_train_landmarks'][:]
    h5f.close()

    d = DataAugmentation(train_landmarks, train_labels)
    d.augmentation(flip=True, noise_levels=[2, 3, 4], rotation=[5, -10, 15])


if __name__ == '__main__':
    main()