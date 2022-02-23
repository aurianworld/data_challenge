from data_augmentation import DataAugmentation
from sklearn.model_selection import train_test_split
import h5py
import pickle
import numpy as np


def write(path, landmarks, labels):
    data = {"landmarks": landmarks, "labels": labels}

    print(data["landmarks"].shape)
    print(data["labels"].shape)
    with open(path, "wb") as file_out:
        pickle.dump(data, file_out)

    print(f"-> {path}")


def main(filename, ratio, data_augmentation=False):

    # get landmarks and labels
    h5f = h5py.File('./dataset/data_train_labels.h5', 'r')
    labels = h5f['data_train_labels'][:]
    h5f.close()
    h5f = h5py.File('./dataset/data_train_landmarks.h5', 'r')
    landmarks = h5f['data_train_landmarks'][:]
    h5f.close()

    h5f = h5py.File('./dataset/data_test_landmarks.h5', 'r')
    landmarks_check = h5f['data_test_landmarks'][:]
    h5f.close()
    labels_check = np.zeros_like(landmarks_check)

    lmk_train, lmk_test, labels_train, labels_test = train_test_split(
        landmarks, labels, test_size=0.20, random_state=42)

    print("Train size (before data augmentation): ", len(lmk_train))
    print("Test size: ", len(lmk_test))

    if data_augmentation:
        d = DataAugmentation(lmk_train, labels_train)
        lmk_train, labels_train = d.augmentation(flip=True,
                                                 noise_levels=[1, 2, 3, 4, 5],
                                                 rotation=[
                                                     -2,
                                                     -1,
                                                     1,
                                                     10,
                                                     -10,
                                                     2,
                                                     5,
                                                     -5,
                                                 ])
    print("Train size (after data augmentation): ", len(lmk_train))
    print("Export data:")

    idx = np.random.permutation(len(lmk_train))
    X_train, y_train = lmk_train[idx], labels_train[idx]

    idx = np.random.permutation(len(lmk_test))
    X_test, y_test = lmk_test[idx], labels_test[idx]

    train_path = "dataset/" + filename + "_train.pickle"
    test_path = "dataset/" + filename + "_test.pickle"
    check_path = "dataset/" + filename + "_check.pickle"

    # write(train_path, X_train, y_train)
    # write(test_path, X_test, y_test)
    write(check_path, landmarks_check, labels_check)

    print(landmarks_check.shape)
    print(labels_check.shape)


if __name__ == '__main__':
    main("dataset", 0.20, True)