import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
from neural_network_utils import *
from inception_blocks_v2 import *
from imutils import paths

MARGIN=96

def triplet_loss(y_true, y_pred, alpha = 0.2):
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss

def build_model():

    K.set_image_data_format('channels_first')
    #np.set_printoptions(threshold=np.nan)
    FRmodel = faceRecoModel(input_shape=(3, MARGIN, MARGIN))

    FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
    load_weights_from_FaceNet(FRmodel)
    return FRmodel

def train(directory, model):
    database = {}
    image_paths = list(paths.list_images(directory))
    for image_path in image_paths:
        label = image_path.split('/')[-2]
        enc = img_to_encoding(image_path, model, MARGIN)
        if enc is not None:
            if label in database:
                database[label].append(enc)
            else:
                database[label] = [enc]

    return database


def predict(directory, database, model):
    predicts = []
    image_paths = list(paths.list_images(directory))
    for image_path in image_paths:
        ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above.
        encoding = img_to_encoding(image_path, model, MARGIN)
    
        ## Step 2: Find the closest encoding ##
        # Initialize "min_dist" to a large value, say 100
        min_dist = 100
    
        # Loop over the database dictionary's names and encodings.
        for (name, db_encs) in database.items():
        
            # Compute L2 distance between the target "encoding" and the current "emb" from the database.
            for enc in db_encs:
                dist = np.linalg.norm(np.subtract(encoding, enc))

                # If this distance is less than the min_dist, then set min_dist to dist, and identity to name.
                if dist < min_dist:
                    min_dist = dist
                    identity = name
    
        if min_dist > 0.7:
            print("Not in the database.")
            predicts.append("unknown")
        else:
            predicts.append(str(identity)) # str(min_dist))
        
    return predicts
    


