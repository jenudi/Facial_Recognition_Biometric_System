import cv2 as cv
import keras
import numpy as np
import os #enables getting file names from directory
import random
import copy
import face_recognition
import pymongo
import bson
#from PIL import Image #we may also use skimage
#from skimage import io #enables reading a single image
#import dlib
#import face_recognition #run this only after installing dlib and face_recognition (above code)
#import tensorflow as tf


#classes for train set images, validation set images and test set images
class image:

    def __init__(self,path):
        self.values=cv.imread(path)
        self.path=path
        self.dir=self.path.split('/')[-2]
        self.person=(' ').join(self.dir.split('_'))
        self.file_name=self.path.split('/')[-1]
        self.in_db=False
        self.face=[]
        self.normalized_values=[]
        self.face_detected=False

    def save(self,new_path,remove_old=False):
        if remove_old:
            os.remove(self.path)
        self.path=new_path
        cv.imwrite(self.path,self.values)

    def detect_face(self):
        face_loc=face_recognition.face_locations(self.values)
        try:
            self.face=self.values[face_loc[0][0]:face_loc[0][1],face_loc[0][3]:face_loc[0][2]]
        except IndexError:
            print("No face detected for " + self.person)
        if len(self.face)>0:
            self.face_detected=True

    def normalize(self):
        self.normalized_values = (self.values - train_image.get_train_mean()) / train_image.get_train_std()

    def preprocess(self):
        self.values = cv.resize(self.values, (256, 256))

    def update_in_db(self,in_db):
        self.in_db=in_db

#all the train set images are saved in a list which is a class variable. this list is used in order or extract the mean and std of the
#train set images for normalization for the rest of the images
class train_image(image):

    train_list=[]

    def __init__(self,path):
        image.__init__(self,path)

        train_image.train_list.append(self)

    @classmethod
    def get_train_mean(cls):
        if len(cls.train_list)>0:
            return np.mean([train_im.values for train_im in cls.train_list],axis=(0,1,2))
        else:
            raise ValueError("No train images")

    @classmethod
    def get_train_std(cls):
        if len(cls.train_list)>0:
            return np.std([train_im.values for train_im in cls.train_list],axis=(0,1,2))
        else:
            raise ValueError("No train images")


class augmentation_image(image):

    augmentation_list=[]

    def __init__(self,path):
        image.__init__(self,path)

        augmentation_image.augmentation_list.append(self)


class validation_image(image):

    validation_list=[]

    def __init__(self,path):
        image.__init__(self,path)

        validation_image.validation_list.append(self)


class test_image(image):

    test_list=[]

    def __init__(self,path):
        image.__init__(self,path)

        test_image.test_list.append(self)



#filters and noise are randomly added to the augmentation images
def identity_filter(image):
  kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
  return cv.filter2D(image, -1, kernel)

def averageing_filter(image):
    return cv.blur(image,(5,5))

def gaussian_filter(image):
    return cv.GaussianBlur(image,(5,5),2)

def median_filter(image):
    return cv.medianBlur(image,5)

def bileteral_filter(image):
    return cv.bilateralFilter(image,5,125,100)

def laplacian_filter(image):
  kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
  return cv.filter2D(image, -1, kernel)

def add_filter(image):
  filters ={0:identity_filter, 1:identity_filter, 2:averageing_filter,
            3:gaussian_filter, 4:median_filter,
            5:bileteral_filter, 6:laplacian_filter}
  return filters[random.randint(0,6)](image)



def salt_and_paper_noise(image):

    num_salt = np.ceil(0.05 * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    image[tuple(coords)] = 1

    num_pepper = np.ceil(0.05 * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    image[tuple(coords)] = 0

    return image

def gaussian_noise(image):
  gauss = np.random.normal(0,0.1**0.5,image.shape)
  return image+gauss.reshape(image.shape)

def poisson_noise(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    return np.random.poisson(image * vals) / float(vals)

def add_noise(image):
   #noises={0:identity_filter, 1:salt_and_paper_noise, 2:gaussian_noise,
            #3:poisson_noise}
   #return noises[random.randint(0,3)](image)
    noises={0:identity_filter, 1:salt_and_paper_noise, 2:gaussian_noise}
    return noises[random.randint(0,2)](image)


def preprocessing_for_augmentation(image):
    image=cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image=add_filter(image)
    image=add_noise(image)
    return image



datagen = keras.preprocessing.image.ImageDataGenerator(
    samplewise_center = True,
    brightness_range=(0.5, 1.5),
    zoom_range = 0.2,
    shear_range = 0.2,
    rotation_range = 30, #Random rotation between 0-30 degrees
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'reflect', #may also try nearest, constant, reflect, wrap. when using 'constant' we should add 'cval' value of 125
    preprocessing_function=preprocessing_for_augmentation)



dataset_dir = 'C:/Users/gash5/Desktop/dataset'
os.chdir(dataset_dir)
directories = [dir for dir in os.listdir(dataset_dir) if not '.' in dir] #directories contain all the people that have images in the dataset

#new direstories are being made for the train, validation and test sets
train_dir = dataset_dir + '/train'
validation_dir = dataset_dir + '/validation'
test_dir = dataset_dir + '/test'
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)
if not os.path.isdir(validation_dir):
    os.mkdir(validation_dir)
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)

for dir in directories:
    files = os.listdir(dataset_dir + '/' + dir)
    images=[file for file in files if ((len(file.split('.')) == 2) and (file.split('.')[1] in ['jpg', 'jpeg', 'png']))] #contains all the images of the person in the current directory
    if len(images)<3:
        continue

    #if a person has less than 3 images he will not be part of any set
    #if a person has 3 images all the images will be put in the train set
    #if a person has 4 images all the images except one will he put in the train set and the last one will be put on the validation set
    #if a person has more than 4 images  all the images except two will he put in the train set and the last two will be put on the validation set and test set
    if not os.path.isdir(train_dir + '/' + dir):
        os.mkdir(train_dir + '/' + dir)

    validation_set = []
    test_set = []
    if len(images)==3:
        train_set=images
    elif len(images)==4:
        train_set=images[:-1]
        validation_set.append(images[-1])
    elif len(images)>4:
        train_set=images[:-2]
        validation_set.append(images[-2])
        test_set.append(images[-1])

#every image that goes to the train set generates 5 new augmentad images
#the image face locations are saved by the method detect_face
    for image_name in train_set:
        new_train_image=train_image(dataset_dir + '/' + dir + '/' + image_name)
        new_train_image.preprocess()
        new_train_image.detect_face()
        new_train_image.save(train_dir + '/' + dir + '/' + image_name)

        image_for_aug = new_train_image.values.reshape((1,) + new_train_image.values.shape)
        i=0
        for batch in datagen.flow(image_for_aug,
                                    batch_size=5,
                                    save_to_dir=train_dir + '/' + dir,
                                    save_prefix='aug',
                                    save_format='jpg'):
            i += 1
            if i == 5:
                break

    for image_name in validation_set:
        if not os.path.isdir(validation_dir + '/' + dir):
            os.mkdir(validation_dir + '/' + dir)
        new_validation_image = validation_image(dataset_dir + '/' + dir + '/' + image_name)
        new_validation_image.preprocess()
        new_validation_image.detect_face()
        new_validation_image.save(validation_dir + '/' + dir + '/' + image_name)

    for image_name in test_set:
        if not os.path.isdir(test_dir + '/' + dir):
            os.mkdir(test_dir + '/' + dir)
        new_test_image = test_image(dataset_dir + '/' + dir + '/' + image_name)
        new_test_image.preprocess()
        new_test_image.detect_face()
        new_test_image.save(test_dir + '/' + dir + '/' + image_name)


#all the augmented values are collected and go through face detection and normalization like the other images
train_directories = [dir for dir in os.listdir(train_dir) if not '.' in dir]
for dir in train_directories:
    files = os.listdir(train_dir + '/' + dir)
    images=[file for file in files if ((len(file.split('.'))==2) and (file.split('.')[1] in ['jpg', 'jpeg', 'png']) and file.split('_')[0]=='aug') ]

    for cur_image in images:
        new_augmentation_image=augmentation_image(train_dir + '/' + dir + '/' + cur_image)
        new_augmentation_image.detect_face()
        new_augmentation_image.normalize()

#all the images in the train, validation and test sets go through normalization
#the normalization type is standartization that is done by substracting the train set mean and dividing by the train set STD
#the normalized values are calculated and saved by the normalize mothod
for cur_image in sum([train_image.train_list, validation_image.validation_list, test_image.test_list],[]):
    cur_image.normalize()

no_faces=[]
#the images the a face have been detected in them are inserted to the database
train_documents=[]
for cur_image in sum([train_image.train_list,augmentation_image.augmentation_list],[]):
    if cur_image.face_detected:
        train_documents.append(bson.son.SON({
        "values":cur_image.values.tolist(),
        "normalized_values":cur_image.normalized_values.tolist(),
        "face":cur_image.face.tolist(),
        "path":cur_image.path,
        "set":"train"
        }))
        cur_image.update_in_db(True)
    else:
        no_faces.append(cur_image)

validation_documents=[]
for cur_image in validation_image.validation_list:
    if cur_image.face_detected:
        validation_documents.append(bson.son.SON({
        "values":cur_image.values.tolist(),
        "normalized_values":cur_image.normalized_values.tolist(),
        "face":cur_image.face.tolist(),
        "path":cur_image.path,
        "set":"validation"
        }))
        cur_image.update_in_db(True)
    else:
        no_faces.append(cur_image)

test_documents=[]
for cur_image in test_image.test_list:
    if cur_image.face_detected:
        test_documents.append(bson.son.SON({
        "values":cur_image.values.tolist(),
        "normalized_values":cur_image.normalized_values.tolist(),
        "face":cur_image.face.tolist(),
        "path":cur_image.path,
        "set":"test"
        }))
        cur_image.update_in_db(True)
    else:
        no_faces.append(cur_image)

if not os.path.isdir(dataset_dir + '/' + 'no_faces_detected'):
    os.mkdir(dataset_dir + '/' + 'no_faces_detected')
for cur_image in no_faces:
    cur_image.save(dataset_dir + '/' + 'no_faces_detected')

client = pymongo.MongoClient('mongodb://localhost:27017/')
with client:
    db = client.biometric_system
    db.faces.insert_many(train_documents)
    db.faces.insert_many(validation_documents)
    db.faces.insert_many(test_documents)