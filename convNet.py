import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time

NAME = "Cars-vs-dog-cnn-64x2-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs\{}'.format(NAME))

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

x = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("Y.pickle","rb"))
test_images = pickle.load(open("X_val.pickle","rb"))
test_labels = pickle.load(open("Y_val.pickle","rb"))

x = np.array(x)
y = np.array(y)
test_images = np.array(test_images)
test_labels = np.array(test_labels)


model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape= x.shape[1:]))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10))

model.summary()
#model.compile(loss="binary_crossentropy",optmizer="adam",metrics=['accuracy'] ,use_multiprocessing=True)
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

datagen = ImageDataGenerator(width_shift_range=0.1, horizontal_flip=True)
datagen.fit(x)

history = model.fit(datagen.flow(x,y),epochs=50 , validation_data=(test_images, test_labels), steps_per_epoch=1, callbacks=[tensorboard])

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.1, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

model.save('CNN.model')