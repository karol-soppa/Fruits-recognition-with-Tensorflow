from data import load_dataset
from neural_network import neural_network
import tensorflow as tf

path_train_img = '.\\Dataset\\train\\images'
path_train_labels = '.\\Dataset\\train\\labels'
path_test_img = '.\\Dataset\\test\\images'
path_test_labels = '.\\Dataset\\test\\labels'
img_train, lab_train = load_dataset(path_train_img, path_train_labels)
img_test, lab_test = load_dataset(path_test_img, path_test_labels)


model = neural_network()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(img_train, lab_train, epochs=35,
                    validation_data=(img_test, lab_test))