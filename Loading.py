import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import random

DATADIR = "D:/Work_Code/aith/train"
CATEGORIES = ["covid", "normal", "pneumonia"]

training_data = []
IMG_SIZE = 50

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
create_training_data()
print(len(training_data))

random.shuffle(training_data)

x = []
y = []

for features, label in training_data:
    x.append(features)
    y.append(label)
x = np.array(x).reshape(-1, IMG_SIZE,IMG_SIZE, 1)

pickle_out = open("X.pickle","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("Y.pickle","rb")
X = pickle.load(pickle_in)

print(X[:10])