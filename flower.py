def recognize_flower(filepath):
    #filename = filepath.split("/")[-1]
    flower_name = name(filepath)
    return flower_name

def get_wikipedia_link(flower):
    wikipedia_link="https://en.m.wikipedia.org/wiki/"+flower
    return wikipedia_link

# For Data Processing & ML Models
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix as cm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
#from tensorflow.keras.losses import *
from tensorflow.keras.models import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image, ImageEnhance
from tensorflow.keras import models


# Miscellaneous
import os
import random

data_dir = 'new/'
unique_labels = os.listdir(data_dir) #list of classes
print(unique_labels)

d = {"21": "fire lily", 
     "3": "canterbury bells", 
     "45": "bolero deep blue", 
     "1": "pink primrose", 
     "34": "mexican aster", 
     "27": "prince of wales feathers", 
     "7": "moon orchid", 
     "16": "globe-flower", 
     "25": "grape hyacinth", 
     "26": "corn poppy", 
     "79": "toad lily", 
     "39": "siam tulip", 
     "24": "red ginger", 
     "67": "spring crocus", 
     "35": "alpine sea holly", 
     "32": "garden phlox", 
     "10": "globe thistle", 
     "6": "tiger lily", 
     "93": "ball moss", 
     "33": "love in the mist", 
     "9": "monkshood", 
     "102": "blackberry lily", 
     "14": "spear thistle", 
     "19": "balloon flower", 
     "100": "blanket flower", 
     "13": "king protea", 
     "49": "oxeye daisy", 
     "15": "yellow iris", 
     "61": "cautleya spicata", 
     "31": "carnation", 
     "64": "silverbush", 
     "68": "bearded iris", 
     "63": "black-eyed susan", 
     "69": "windflower", 
     "62": "japanese anemone", 
     "20": "giant white arum lily", 
     "38": "great masterwort", 
     "4": "sweet pea", 
     "86": "tree mallow", 
     "101": "trumpet creeper", 
     "42": "daffodil", 
     "22": "pincushion flower", 
     "2": "hard-leaved pocket orchid", 
     "54": "sunflower", 
     "66": "osteospermum", 
     "70": "tree poppy", 
     "85": "desert-rose", 
     "99": "bromelia", 
     "87": "magnolia", 
     "5": "english marigold", 
     "92": "bee balm", 
     "28": "stemless gentian", 
     "97": "mallow", 
     "57": "gaura", 
     "40": "lenten rose", 
     "47": "marigold", 
     "59": "orange dahlia", 
     "48": "buttercup", 
     "55": "pelargonium", 
     "36": "ruby-lipped cattleya", 
     "91": "hippeastrum", 
     "29": "artichoke", 
     "71": "gazania", 
     "90": "canna lily", 
     "18": "peruvian lily", 
     "98": "mexican petunia", 
     "8": "bird of paradise", 
     "30": "sweet william", 
     "17": "purple coneflower", 
     "52": "wild pansy", 
     "84": "columbine", 
     "12": "colt's foot", 
     "11": "snapdragon", 
     "96": "camellia", 
     "23": "fritillary", 
     "50": "common dandelion", 
     "44": "poinsettia", 
     "53": "primula", 
     "72": "azalea", 
     "65": "californian poppy", 
     "80": "anthurium", 
     "76": "morning glory", 
     "37": "cape flower", 
     "56": "bishop of llandaff", 
     "60": "pink-yellow dahlia", 
     "82": "clematis", 
     "58": "geranium", 
     "75": "thorn apple", 
     "41": "barbeton daisy", 
     "95": "bougainvillea", 
     "43": "sword lily", 
     "83": "hibiscus", 
     "78": "lotus lotus", 
     "88": "cyclamen", 
     "94": "foxglove", 
     "81": "frangipani", 
     "74": "rose", 
     "89": "watercress", 
     "73": "water lily", 
     "46": "wallflower", 
     "77": "passion flower", 
     "51": "petunia"}

IMAGE_SIZE = 128 #to resize image to a specific dimension

model = models.load_model("flower102save")


from keras.preprocessing import image

def name(filename):
    test_image = tf.keras.utils.load_img(filename,target_size=(IMAGE_SIZE,IMAGE_SIZE))
    test_image = tf.keras.utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    result = model.predict(test_image)

    pred = np.argmax(result) # get the index of max value
    return d[unique_labels[pred]]