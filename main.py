import os

import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image

def img_dataset_from_directory(path):
    data = []
    for filename in os.listdir(path):
        sth = Image.open(os.path.join(path, filename))
        data.append(np.array(sth))
    data = np.array(data)
    return data

def label_dataset_from_directory(path):
    data = []
    for filename in os.listdir(path):
        sth = Image.open(os.path.join(path, filename))
        df = pd.read_csv(os.path.join(path, filename), sep=" ", header=None)
        a = df.iloc[0, 0]
        data.append(np.array(a))
    data = np.array(data)
    return data


path_train_img = '.\\Dataset\\test\\images'
images_train = []
for filename in os.listdir(path_train_img):
    img = Image.open(os.path.join(path_train_img, filename))
    images_train.append(np.array(img))
images_train=np.array(images_train)
path_train_labels = '.\\Dataset\\test\\labels'
labels_train = []
for filename in os.listdir(path_train_labels):
    df = pd.read_csv(os.path.join(path_train_labels, filename), sep=" ", header=None)
    a = df.iloc[0, 0]
    labels_train.append(np.array(a))
labels_train = np.array(labels_train)


print("Format train_images:", type(images_train), images_train.shape)
print("Format train_labels:", type(labels_train), labels_train.shape)
