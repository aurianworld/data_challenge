from data_augmentation import DataAugmentation
from sklearn.model_selection import train_test_split
import h5py
import pickle


def write(path, landmarks, labels):
    data = {"landmarks": landmarks, "labels": labels}

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

    lmk_train, lmk_test, labels_train, labels_test = train_test_split(
        landmarks, labels, test_size=0.20, random_state=42)

    print("Train size (before data augmentation): ", len(lmk_train))
    print("Test size: ", len(lmk_test))

    if data_augmentation:
        d = DataAugmentation(lmk_train, labels_train)
        lmk_train, labels_train = d.augmentation(flip=True,
                                                 noise_levels=[2, 3, 4],
                                                 rotation=[
                                                     -2,
                                                     2,
                                                     5,
                                                     -5,
                                                 ])
    print("Train size (after data augmentation): ", len(lmk_train))
    print("Export data:")

    train_path = "dataset/" + filename + "_train.pickle"
    test_path = "dataset/" + filename + "_test.pickle"

    write(train_path, lmk_train, labels_train)
    write(test_path, lmk_test, labels_test)


if __name__ == '__main__':
    main("dataset", 0.20, True)