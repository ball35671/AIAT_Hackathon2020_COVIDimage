import cv2
import tensorflow as tf

CATEGORIES = ["covid", "normal", "pneumonia"]

def prepare(filepath):
    IMG_SIZE = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_array = img_array/255.0
    new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE,IMG_SIZE, 1)

model = tf.keras.models.load_model("CNN.model")

prediction = model.predict([prepare('normal/normal001.png')])
print(CATEGORIES[int(prediction[0][0])])