import cv2 as cv
import numpy as np
import pandas as pd
import os #enables getting file names from directory
import random
import face_recognition #run this only after installing dlib, cmake and face_recognition
import pymongo
import bson
from sets_splits import db_df


if __name__ == "__main__":

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