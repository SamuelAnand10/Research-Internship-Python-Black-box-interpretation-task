# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:02:29 2023

@author: samue
"""

import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#%%
from keras.applications.vgg16 import VGG16, preprocess_input
model = VGG16(weights='imagenet')

#%%
import requests
import json
url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
response = requests.get(url)
CLASS_INDEX = json.loads(response.content.decode())
classlabel = []
for i_dict in range(len(CLASS_INDEX)):
    classlabel.append(CLASS_INDEX[str(i_dict)][1])
print("N of class = {}".format(len(classlabel)))
#%%
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

_img = load_img('C:/Users/samue/Downloads/dog_and_cat.jpg', target_size=(224, 224))
plt.imshow(_img)
plt.show()


#%%

#%%
