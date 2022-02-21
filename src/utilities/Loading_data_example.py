import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2

#######################################################################
# Loading data tables from H5py files
#######################################################################

h5f = h5py.File('.\dataset\data_train_images.h5','r')
train_images = h5f['data_train_images'][:]
h5f.close()

h5f = h5py.File('.\dataset\data_train_landmarks.h5','r')
train_landmarks = h5f['data_train_landmarks'][:]
h5f.close()

h5f = h5py.File('.\dataset\data_train_labels.h5','r')
train_labels = h5f['data_train_labels'][:]
h5f.close()

#######################################################################
# display the size of the different data tables
#######################################################################

print("Images")
print(train_images.shape)
print(train_images[0].shape)
print(train_images[0][0].shape)
print("\nLandmarks")
print(train_landmarks.shape)
print(train_landmarks[0].shape)
print(train_landmarks[0][0].shape)
print("\nLabels")
print(train_labels.shape)

#######################################################################
# display an example by giving the sequence number ( Train in [0,540[ ) 
# and the sequence frame number ( in [0,9] )
#######################################################################

sequence = 127
frame = 0

img = train_images[sequence,frame]
landmarks = train_landmarks[sequence,frame]

expressions = ['happiness','fear','surprise','anger','disgust','sadness']

print("Label (sequence : " + str(sequence) + ") = " + str(train_labels[sequence]) + " -> " + expressions[int(train_labels[sequence])])

for p in landmarks:
    img = cv2.circle(img, (p[0],p[1]), 2, (255,255,255), 2)

plt.imshow(img)
plt.show()