from __future__ import division, print_function

import sys
import os
import glob
import re
import numpy as np
import json


from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
import keras.utils as image
import tensorflow as tf
from keras.models import Sequential
from keras_preprocessing.image import load_img

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)




@app.route('/', methods=['GET'])
def index():

    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():


    def model_predict(img_path, model):
        img = load_img(img_path, target_size=(180, 180))

        x = image.img_to_array(img)

        x = np.expand_dims(x, axis=0)

        x = preprocess_input(x, mode='caffe')

        preds = model.predict(x)
        return preds


    def read_image(filename):
        img = load_img(filename, target_size=(180, 180))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    def bubble_sort(array):
        n = len(array[0])
        for i in range(n - 1):
            for j in range(n - i - 1):
                if array[0][j][2] > array[0][j + 1][2]:
                    array[0][j], array[0][j + 1] = array[0][j + 1], array[0][j]
        return array

    def minus(a):
        if a < 0:
            result = a
        else:
            result = 2 * a
        return result


    if request.method == 'POST':
        model = load_model('/home/woorym/Similarface/model/model5.h5')
        model.make_predict_function()

        def decode(preds, top=5, class_list_path='/home/woorym/Similarface/index/index.json'):
            if len(preds.shape) != 2 or preds.shape[1] != 6: # your classes number
                raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
            index_list = json.load(open(class_list_path))
            results = []
            for pred in preds:
                top_indices = pred.argsort()[-top:][::-1]
                result = [tuple(index_list[str(i)]) + (pred[i],) for i in top_indices]
                result.sort(reverse=True)
                results.append(result)
            return results
        f = request.files['file']


        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads/new', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)

        pred_class = decode(preds)

        num = bubble_sort(pred_class)

        result2 = '''1. {} {:.4f}%   '''.format(num[0][4][1],2 * num[0][4][2])
        other1 = '''2. {} {:.4f}%   '''.format(pred_class[0][3][1],2 * pred_class[0][3][2])
        other2 = '''3. {} {:.4f}%   '''.format(pred_class[0][2][1],minus(pred_class[0][2][2]))


        final = '''{}  {}  {}'''.format(result2, other1, other2)


        return final
    return None




if __name__ == "__main__":
    app.run(debug=False,port = 6000)