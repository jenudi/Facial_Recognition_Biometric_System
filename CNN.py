import cv2 as cv
import keras
import numpy as np
import pandas as pd
import os #enables getting file names from directory
import random
import face_recognition #run this only after installing dlib, cmake and face_recognition
import pymongo
import bson
from preprocessing import image, train_set, validation_set, test_set

#all the images in the train, validation and test sets go through normalization
#the normalization type is standardization that is done by substracting the train set mean and dividing by the train set STD
#the normalized values are calculated by the normalize mothod

train_set_df=pd.DataFrame(columns=['normalized values', 'label'])

for index,image_name in enumerate(train_set):
    cur_image = image(image_name)
    cur_image.values = cur_image.detect_face()
    if not (cur_image.values is None):
        train_set_df.loc[index]=[cur_image.normalize(),cur_image.person]

validation_set_df=pd.DataFrame(columns=['normalized values', 'label'])

for index,image_name in enumerate(validation_set):
    cur_image = image(image_name)
    cur_image.values = cur_image.detect_face()
    if not (cur_image.values is None):
        validation_set_df.loc[index]=[cur_image.normalize(),cur_image.person]

test_set_df=pd.DataFrame(columns=['normalized values', 'label'])

for index,image_name in enumerate(test_set):
    cur_image = image(image_name)
    cur_image.values = cur_image.detect_face()
    if not (cur_image.values is None):
        test_set_df.loc[index]=[cur_image.normalize(),cur_image.person]



'''''
train_documents=[]
for cur_image in sum([train_image.train_paths_list,augmentation_image.augmentation_paths_list],[]):
    cur_train_image=image(cur_image)
    cur_train_image.values=cur_train_image.detect_face()
    normalized_values=cur_train_image.normalize()
    train_documents.append(bson.son.SON({
    "values":cur_train_image.values.tolist(),
    "normalized_values":normalized_values.tolist(),
    "person":cur_train_image.person,
    "path":cur_train_image.path,
    "set":"train"
    }))

validation_documents=[]
for cur_image in validation_image.validation_paths_list:
    cur_validation_image=image(cur_image)
    cur_validation_image.values=cur_validation_image.detect_face()
    normalized_values=cur_validation_image.normalize()
    validation_documents.append(bson.son.SON({
    "values":cur_train_image.values.tolist(),
    "normalized_values":normalized_values.tolist(),
    "person":cur_train_image.person,
    "path":cur_validation_image.path,
    "set":"validation"
    }))

test_documents=[]
for cur_image in test_image.test_paths_list:
    cur_test_image=image(cur_image)
    cur_test_image.values=cur_test_image.detect_face()
    normalized_values=cur_test_image.normalize()
    test_documents.append(bson.son.SON({
    "values":cur_train_image.values.tolist(),
    "normalized_values":normalized_values.tolist(),
    "person":cur_train_image.person,
    "path":cur_test_image.path,
    "set":"test"
    }))


client = pymongo.MongoClient('mongodb://localhost:27017/')
with client:
    db = client.biometric_system
    db.faces.insert_many(train_documents)
    db.faces.insert_many(validation_documents)
    db.faces.insert_many(test_documents)
'''''

