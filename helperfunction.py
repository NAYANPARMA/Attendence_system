
from keras.models import load_model
import cv2
import sys
import tensorflow as tf
import numpy as np
import keras

def triplet_loss(y_true, y_pred, alpha = 0.2):    
    anchor,position,nagative=y_pred[:,0:128],y_pred[:,128:256],y_pred[:,256:384]
    
    pos_dist =tf.reduce_sum(tf.square(tf.subtract(anchor,position)),axis=-1)
    neg_dist =  tf.reduce_sum(tf.square(tf.subtract(anchor,nagative)),axis = -1)
    basic_loss = tf.add(tf.subtract(pos_dist,neg_dist),alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss,0.0),axis = None)
    
    return loss

    
def image_resizing(image):

    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    classifier=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces=classifier.detectMultiScale(gray,1.1,6)

    x,y,w,h=faces.squeeze()
    crop=image[y:y+h,x:x+w]
    image=cv2.resize(crop,(96,96))
    
    return image

def face(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    image = cv2.resize(image, (250,250))    
    return image

def img_to_encoding(image, model):
    img = image[...,::-1]
    img = np.around(img/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.layers[3].predict_on_batch(x_train)
    return embedding

threshold=0.65
interval=0.3
def computeconfidence(ref_encode,img_encode,thres=threshold):
    dist=np.linalg.norm((img_encode-ref_encode))
    confidence=(threshold-max([dist,interval]))/(threshold-interval)
    return dist,confidence