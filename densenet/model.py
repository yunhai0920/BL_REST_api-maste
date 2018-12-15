# -*- coding:utf-8 -*-
import os
import numpy as np
from imp import reload
from PIL import Image, ImageOps
import io
from keras.layers import Input
from keras.models import Model

# import keras.backend as K

from . import densenet

reload(densenet)

char_file = './char_std_4944.txt'
characters = io.open(char_file, 'r', encoding='utf-8').readlines()
characters = characters[0].split()+['卍']
nclass = len(characters)
inputs = Input(shape=(32, None, 1), name='the_input')
y_pred = densenet.dense_cnn(inputs, nclass)
basemodel = Model(inputs=inputs, outputs=y_pred)
modelPath = os.path.join(os.getcwd(), 'densenet/models/weights_densenet.h5')
if os.path.exists(modelPath):
    basemodel.load_weights(modelPath)


def decode(pred):
    char_list = []
    pred_text = pred.argmax(axis=2)[0]
    for i in range(len(pred_text)):
        if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
            char_list.append(characters[pred_text[i]])
    return u''.join(char_list)


def predict(img):
    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / 32
    width = int(width / scale)
    
    img = img.resize([width, 32], Image.ANTIALIAS)

    img = np.array(img).astype(np.float32) / 255.0 - 0.5
    
    x = img.reshape([1, 32, width, 1])
    y_preds = basemodel.predict(x)
    y_preds = y_preds[:, :, :]

    out = decode(y_preds)
    return out
