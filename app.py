import numpy as np
from numpy.core.arrayprint import str_format
import requests
from flask import Flask,jsonify,request,json
import flask
import jsonpickle
import werkzeug
from PIL import Image
from PIL.Image import Image
import requests
import json
import cv2
import numpy as np
import jsonpickle
import urllib.request
from PIL import Image

import urllib3
from urllib3.packages.six import StringIO

import tensorflow.keras
from PIL import Image, ImageOps
import datetime
import time
# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
# import seaborn as sns
# %matplotlib inline
import warnings
style.use('fivethirtyeight')


from tensorflow import keras
#preprocess.
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#dl libraraies
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam,SGD,RMSprop
from tensorflow.keras.utils import to_categorical

# specifically for cnn
from tensorflow.keras.layers import Dropout, Flatten,Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import InputLayer
 
import tensorflow as tf
import random as rn

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
import numpy as np  
# from tqdm import tqdm
import os                   
from random import shuffle  
from PIL import Image
import tensorflow.keras.preprocessing.image as img
from urllib.parse import quote, urlparse, urlunparse

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
size = (224, 224)
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


model = tensorflow.keras.models.load_model("TM450New.h5")

output1=[]
ima1=""
app = flask.Flask(__name__)

@app.route('/')
def predict():
        ima=request.args.get('ima',type=urlparse)
        ima1=urlunparse(ima)
        print("image:"  )
        print(ima1)
        
        
        

        urllib.request.urlretrieve( str(ima1) ,"gfg.png")

        img = Image.open("gfg.png")
        if (img.size[0]>img.size[1]):
         img = img.rotate(-90, expand=True)

        img=np.asarray(img)

        crop1=img[0:100,0:380]
        Oimg=[]

        Oimg.append(crop1[0:crop1.shape[1],0:55])
        Oimg.append(crop1[0:crop1.shape[1],55:95])
        Oimg.append(crop1[0:crop1.shape[1],95:140])
        Oimg.append(crop1[0:crop1.shape[1],140:180])
        Oimg.append(crop1[0:crop1.shape[1],180:230])
        Oimg.append(crop1[0:crop1.shape[1],230:280])
        Oimg.append(crop1[0:crop1.shape[1],280:320])
        Oimg.append(crop1[0:crop1.shape[1],320:370])
        output1=[]
        for i in Oimg:
                X = increase_brightness(i, value=80)
                X=Image.fromarray(np.uint8(X))
                X = ImageOps.fit(X, size, Image.ANTIALIAS)
                
                X = np.array(X)
                normalized_image_array = (X.astype(np.float32) / 127.0) - 1
                data[0] = normalized_image_array
                prediction = model.predict(data)
                j=0
                print(prediction[0].tolist().index(prediction[0].max()))
                output=prediction[0].tolist().index(prediction[0].max())
                output1.append(output)
                
         
        return  jsonify(output1)





if __name__ == '__main__':
    
    app.run()





       
