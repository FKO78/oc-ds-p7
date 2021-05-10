"""
Prédiction de'une espèce de chien à partir de sa photo (array).
"""

import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageOps
#from myfunctions import *
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
import pickle

def clean_img(imgarr, w, h):
    """
    Fonction de nettoyage/formatage (w*h) d'une array d'une image RGB pour prédiction VGG16
    """

    # Correction de l'exposition
    img_conv = ImageOps.autocontrast(Image.fromarray(imgarr))

    # Correction du contraste
    img_conv = ImageOps.equalize(img_conv)

    # Normalisation et application d'un filtre médian
    img_conv = cv2.medianBlur(np.array(img_conv), 3)

    # Dimensionnement des images
    return cv2.cvtColor(cv2.resize(img_conv, (w, h)), cv2.COLOR_BGR2RGB)

mymodel = load_model('OC_DS_P7_mymodel.h5')

with open('OC_DS_P7_galerie_encoder.pkl', 'rb') as file:
    unpickler = pickle.Unpickler(file)
    galerie_single = unpickler.load()
    breed_enc = unpickler.load()

os.system("cls")

fichier = input("Quel est le nom du fichier image ? ")
imgarr = cv2.imread(str(fichier))

img = clean_img(imgarr, 224, 224)
img_vgg = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))  # Créer la collection d'images (un seul échantillon)
img_vgg = preprocess_input(img_vgg)  # Prétraiter l'image comme le veut VGG-16
idx = np.argmax(mymodel.predict(img_vgg))
breed = breed_enc.categories_[0][idx]

os.system("cls")
print('Espèce {} détectée sur {}'.format(breed, fichier))
