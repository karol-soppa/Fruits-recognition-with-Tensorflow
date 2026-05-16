from data import img_dataset_from_directory
from data import label_dataset_from_directory
from neural_network import neural_network
import tensorflow as tf

path_train_img = '.\\Dataset\\train\\images'
path_train_labels = '.\\Dataset\\train\\labels'
path_test_img = '.\\Dataset\\test\\images'
path_test_labels = '.\\Dataset\\test\\labels'
img_train = img_dataset_from_directory(path_train_img)
lab_train = label_dataset_from_directory(path_train_labels)
img_test = img_dataset_from_directory(path_test_img)
lab_test = label_dataset_from_directory(path_test_labels)

model = neural_network()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(img_train, lab_train, epochs=35,
                    validation_data=(img_test, lab_test))