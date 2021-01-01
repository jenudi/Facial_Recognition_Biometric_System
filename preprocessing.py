import numpy.core.multiarray
import cv2 as cv
import keras
import numpy as np
import os #enables getting file names from directory
import random
import face_recognition #run this only after installing dlib, cmake and face_recognition
import pymongo
import bson
import math


#classes for train set images, validation set images and test set images
class image:

    def __init__(self,path):
        self.values=cv.imread(path)
        self.path=path
        self.dir=self.path.split('/')[-2]
        self.person=(' ').join(self.dir.split('_'))
        self.file_name=self.path.split('/')[-1]

    def save(self,new_path,remove_old=False):
        if remove_old:
            os.remove(self.path)
        self.path=new_path
        cv.imwrite(self.path,self.values)

    def detect_face(self):
        face_loc=face_recognition.face_locations(self.values)
        try:
            face=self.values[face_loc[0][0]:face_loc[0][1],face_loc[0][3]:face_loc[0][2]]
        except IndexError:
            return None
        if (not face is None) and (len(face)>0):
            return face
        else:
            return None

    def normalize(self):
        return (self.values - train_image.get_train_mean()) / train_image.get_train_std()

    def preprocess(self):
        self.values = cv.resize(self.values, (256, 256))


#all the train set images are saved in a list which is a class variable. this list is used in order or extract the mean and std of the
#train set images for normalization for the rest of the images
class train_image(image):

    train_paths_list=[]

    def __init__(self,path):
        image.__init__(self,path)

        train_image.train_paths_list.append(path)

    @classmethod
    def get_train_mean(cls):
        if len(cls.train_paths_list)>0:
            train_mean=[]
            for train_path in cls.train_paths_list:
                train_image=image(train_path)
                train_mean.append(train_image.values)
                return np.mean(train_mean,axis=(0,1,2))
        else:
            raise ValueError("No train images")

    @classmethod
    def get_train_std(cls):
        if len(cls.train_paths_list)>0:
            train_std=[]
            for train_path in cls.train_paths_list:
                train_image=image(train_path)
                train_std.append(train_image.values)
                return np.std(train_std,axis=(0,1,2))
        else:
            raise ValueError("No train images")

class augmentation_image(image):

    augmentation_paths_list=[]

    def __init__(self,path):
        image.__init__(self,path)

        augmentation_image.augmentation_paths_list.append(path)


class validation_image(image):

    validation_paths_list=[]

    def __init__(self,path):
        image.__init__(self,path)

        validation_image.validation_paths_list.append(path)


class test_image(image):

    test_paths_list=[]

    def __init__(self,path):
        image.__init__(self,path)

        test_image.test_paths_list.append(path)



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
    if len(images)<2:
        continue

    #if a person has less than 3 images he will not be part of any set
    #if a person has 3 images all the images will be put in the train set
    #if a person has 4 images all the images except one will he put in the train set and the last one will be put on the validation set
    #if a person has more than 4 images  all the images except two will he put in the train set and the last two will be put on the validation set and test set

    train_set=[]
    test_set = []
    validation_set = []
    if len(images)==2:
        train_set.append(images[0])
        test_set.append(images[1])
    elif len(images)==3:
        train_set.append(images[0])
        test_set.append(images[1])
        validation_set.append(images[2])
    elif len(images)>3:
        train_set=images[0:math.floor(len(images)/0.5)]
        test_set=images[math.floor(len(images)/0.5):math.floor(len(images)/0.75)]
        validation_set=images[math.floor(len(images)/0.75):]

#every image that goes to the train set generates 5 new augmentad images
#the image face locations are saved by the method detect_face

    no_faces_detected=[]
    for image_name in train_set:
        if not os.path.isdir(train_dir + '/' + dir):
            os.mkdir(train_dir + '/' + dir)
        new_train_image=train_image(dataset_dir + '/' + dir + '/' + image_name)
        face=new_train_image.detect_face()
        if not (face is None):
            new_train_image.preprocess()
            new_path=train_dir + '/' + dir + '/' + image_name
            train_image.train_paths_list[train_image.train_paths_list.index(new_train_image.path)]=new_path
            new_train_image.save(new_path)

            image_for_aug = new_train_image.values.reshape((1,) + new_train_image.values.shape)
            i=0
            for batch in datagen.flow(image_for_aug,
                                        batch_size=1,
                                        save_to_dir=train_dir + '/' + dir,
                                        save_prefix='aug',
                                        save_format='jpg'):
                i += 1
                if i == 5:
                    break

        else:
            print("No face detected for " + new_train_image.person)
            no_faces_detected.append(new_train_image.path)
            train_image.train_paths_list.remove(new_train_image.path)


    for image_name in validation_set:
        if not os.path.isdir(validation_dir + '/' + dir):
            os.mkdir(validation_dir + '/' + dir)
        new_validation_image = validation_image(dataset_dir + '/' + dir + '/' + image_name)
        face=new_validation_image.detect_face()
        if not (face is None):
            new_validation_image.preprocess()
            new_path=validation_dir + '/' + dir + '/' + image_name
            validation_image.validation_paths_list[validation_image.validation_paths_list.index(new_validation_image.path)]=new_path
            new_validation_image.save(new_path)
        else:
            print("No face detected for " + new_validation_image.person)
            no_faces_detected.append(new_validation_image.path)
            validation_image.validation_paths_list.remove(new_validation_image.path)


    for image_name in test_set:
        if not os.path.isdir(test_dir + '/' + dir):
            os.mkdir(test_dir + '/' + dir)
        new_test_image = test_image(dataset_dir + '/' + dir + '/' + image_name)
        face=new_test_image.detect_face()
        if not (face is None):
            new_path=test_dir + '/' + dir + '/' + image_name
            new_test_image.preprocess()
            test_image.test_paths_list[test_image.test_paths_list.index(new_test_image.path)]=new_path
            new_test_image.save(new_path)
        else:
            print("No face detected for " + new_test_image.person)
            no_faces_detected.append(new_train_image.path)
            test_image.test_paths_list.remove(new_test_image.path)


#all the augmented values are collected and go through face detection and normalization like the other images
train_directories = [dir for dir in os.listdir(train_dir) if not '.' in dir]
for dir in train_directories:
    files = os.listdir(train_dir + '/' + dir)
    augmentation_images=[file for file in files if ((len(file.split('.'))==2) and (file.split('.')[1] in ['jpg', 'jpeg', 'png']) and file.split('_')[0]=='aug') ]

    for cur_image in augmentation_images:
        new_augmentation_image=augmentation_image(train_dir + '/' + dir + '/' + cur_image)
        face=new_augmentation_image.detect_face()
        if (face is None):
            augmentation_image.augmentation_paths_list.remove(new_augmentation_image.path)
            os.remove(new_augmentation_image.path)

normalized_values=[]
labels=[]
whole_train_set=sum([train_image.train_paths_list,augmentation_image.augmentation_paths_list],[])

for set in [whole_train_set,validation_image.validation_paths_list,test_image.test_paths_list]:
    for image_name in set:
        cur_image = image(image_name)
        cur_image.values = cur_image.detect_face()
        normalized_values.append(cur_image.normalize())
        labels.append(cur_image.person)

if not os.path.isdir(dataset_dir + '/' + 'no_faces_detected'):
    os.mkdir(dataset_dir + '/' + 'no_faces_detected')
for image_name in no_faces_detected:
    no_face_image=image(image_name)
    no_face_image.save(dataset_dir + '/' + 'no_faces_detected' + '/' + image_name)

'''''
#all the images in the train, validation and test sets go through normalization
#the normalization type is standartization that is done by substracting the train set mean and dividing by the train set STD
#the normalized values are calculated by the normalize mothod

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

