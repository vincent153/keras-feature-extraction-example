import numpy as np
import cv2
import keras
from keras.models import load_model
from keras.models import Model
import sys


model = None;

#dense_1
def load_mnist_model():
    global model
    model = load_model('mnist_test.h5')
    print(model.summary())

def read_data(file_name):
    img = cv2.imread(file_name)
    col,row = img.shape[:2]
    if col!=28 or row!=28:
        print('resizing img for network')
        img = cv2.resize(img,(28,28))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.reshape(28,28,1)

    return np.array([gray])

def predict(file_name):
    data = read_data(file_name)
    print(model.predict_classes(data))

def feature_extrction(layer_name,file_name):
    data = read_data(file_name)
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(data)
    print(intermediate_output)
    pass

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(0)
    load_mnist_model()
    #predict(sys.argv[1])
    feature_extrction('dense_1',sys.argv[1])
    #feature_extrction('dense_1')
