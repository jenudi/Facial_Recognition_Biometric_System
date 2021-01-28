import os
import math
import pandas as pd
from face_recognition import face_locations
from augmentation import *
import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#%% classes for train set images, validation set images and test set images
class image_in_set:

    name_to_id_dict=dict()

    def __init__(self,path):
        self.values=cv.imread(path)
        self.path=path
        self.dir=self.path.split('\\')[-2]
        self.file_name=self.path.split('\\')[-1]
        self.name=' '.join(self.dir.split('_'))
        self.id=image_in_set.name_to_id(self.name)

    @classmethod
    def name_to_id(cls,name):
        if name in cls.name_to_id_dict.keys():
            return cls.name_to_id_dict[name]
        else:
            new_id=len(cls.name_to_id_dict.keys())+1
            cls.name_to_id_dict[name]=new_id
            return new_id

    def get_face_image(self):
        face_loc=face_locations(self.values)
        try:
            face=self.values[face_loc[0][0]:face_loc[0][1],face_loc[0][3]:face_loc[0][2]]
        except IndexError:
            return None
        if (not face is None) and (not isinstance(face, type(None))) and len(face):
            return face_image(face,self.name)
        else:
            return None

    def normalize_by_train_values(self,train_paths_list):
        train_mean = get_images_mean(train_paths_list)
        train_std = get_images_std(train_paths_list)
        return (self.values - train_mean) / train_std

    def normalize_by_image_values(self):
        return (self.values- self.values.mean())/self.values.std()

    def save(self, new_path):
        self.path = new_path
        cv.imwrite(self.path, self.values)

    def resize_image(self):
        self.values = cv.resize(self.values, (160, 160))


class face_image(image_in_set):

    def __init__(self,values,name):
        self.values=values
        self.name=name
        self.path=None


def get_images_mean(paths_list):
    assert len(paths_list), "paths list must not be empty"
    train_mean=list()
    for path in paths_list:
        train_image=image_in_set(path[0])
        train_mean.append(train_image.values)
    return np.mean(train_mean, axis=(0, 1, 2))

def get_images_std(paths_list):
    assert len(paths_list), "paths list must not be empty"
    train_std=list()
    for path in paths_list:
        train_image=image_in_set(path[0])
        train_std.append(train_image.values)
    return np.std(train_std,axis=(0,1,2))


def get_embedding(cur_image, normalization_method, model, train_paths_list=None):
    if normalization_method=="normalize_by_train_values":
        assert (not train_paths_list is None) or (not isinstance(face, type(None))), "enter train paths list in order to use the normalize by train values method"
        norm_values = cur_image.normalize_by_train_values(train_paths_list).astype("float32")
    else:
        norm_values = cur_image.normalize_by_image_values()
    four_dim_values = np.expand_dims(norm_values, axis=0)
    embedding=model.predict(four_dim_values)[0]
    return embedding


#all the train set images are saved in a list which is a class variable. this list is used in order or extract the mean and std of the
#train set images for normalization for the rest of the images

dataset_dir=input("Please enter the dataset directory path")

number_of_train_images = 100

os.chdir(dataset_dir)
directories = [dir for dir in os.listdir(dataset_dir) if not '.' in dir] #directories contain all the people that have images in the dataset

#new direstories are being made for the train, validation and test sets
sets_dir=''.join([dataset_dir, '\\sets'])
if not os.path.isdir(sets_dir):
    os.mkdir(sets_dir)

train_dir = ''.join([sets_dir, '\\train'])
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)

validation_dir = ''.join([sets_dir, '\\validation'])
if not os.path.isdir(validation_dir):
    os.mkdir(validation_dir)

test_dir = ''.join([sets_dir, '\\test'])
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)

train_paths = list()
validation_paths = list()
test_paths = list()
augmentation_paths = list()
no_faces_detected = list()

for dir in directories:
    dir_path=''.join([dataset_dir, '\\', dir])
    files = os.listdir(dir_path)
    images=[file for file in files if ((len(file.split('.')) == 2) and (file.split('.')[1] in ['jpg', 'jpeg', 'png']))] #contains all the images of the person in the current directory
    if len(images)<2:
        continue


    #if a person has less than 2 images he will not be part of any set
    #if a person has 2 images all the images will be put in the train set and test set
    #if a person has 3 images 1 image will be in the train, validation and test set
    #if a person has more than 3 50 % of the images will be in the train set and 25% and 25% will be in the validation and test set
    for image_name in images:
        cur_image=image_in_set(''.join([dir_path, '\\', image_name]))
        face=cur_image.get_face_image()
        if (face is None) or (isinstance(face, type(None))):
            images.remove(image_name)
            print(' '.join(["No face detected for",cur_image.name]))
            no_faces_detected.append(cur_image.path)

    cur_train_set= list()
    cur_test_set = list()
    cur_validation_set = list()

    if len(images)==2:
        cur_train_set.append(image_in_set(''.join([dir_path, '\\', images[0]])))
        cur_test_set.append(image_in_set(''.join([dir_path, '\\', images[1]])))
    elif len(images)==3:
        cur_train_set.append(image_in_set(''.join([dir_path, '\\', images[0]])))
        cur_validation_set.append(image_in_set(''.join([dir_path, '\\', images[1]])))
        cur_test_set.append(image_in_set(''.join([dir_path, '\\', images[2]])))
    elif len(images)>3:
        cur_train_set=[image_in_set(''.join([dir_path, '\\', cur_image])) for cur_image in images[:math.ceil(len(images)*0.5)]]
        cur_validation_set=[image_in_set(''.join([dir_path, '\\', cur_image])) for cur_image in images[math.ceil(len(images)*0.5):round(len(images)*0.75)]]
        cur_test_set=[image_in_set(''.join([dir_path, '\\', cur_image])) for cur_image in images[round(len(images) * 0.75):]]

    #every image that goes to the train set generates 5 new augmentad images
    #the image face locations are saved by the method get_face_image
    for new_train_image in cur_train_set:
        new_train_dir=''.join([train_dir, '\\', dir])

        if not os.path.isdir(new_train_dir):
            os.mkdir(new_train_dir)
        new_face_image = new_train_image.get_face_image()
        new_face_image.resize_image()
        new_path=''.join([train_dir, '\\', dir, '\\', new_train_image.file_name])
        old_path=new_train_image.path
        new_face_image.save(new_path)
        train_paths.append((new_path, old_path))

        image_for_aug = new_train_image.values.reshape((1,) + new_train_image.values.shape)

        number_of_augmentations=(number_of_train_images-len(cur_train_set))/len(cur_train_set)
        i=0
        for batch in datagen.flow(image_for_aug, batch_size=1, save_to_dir=new_train_dir, save_prefix='aug', save_format='jpg'):
            i += 1
            if i ==number_of_augmentations:
                break

    for new_validation_image in cur_validation_set:
        new_validation_dir=''.join([validation_dir, '\\', dir])
        if not os.path.isdir(new_validation_dir):
            os.mkdir(new_validation_dir)
        new_face_image = new_validation_image.get_face_image()
        new_face_image.resize_image()
        new_path=''.join([validation_dir, '\\', dir, '\\', new_validation_image.file_name])
        old_path=new_validation_image.path
        new_face_image.save(new_path)
        validation_paths.append((new_path, old_path))

    for new_test_image in cur_test_set:
        new_test_dir=''.join([test_dir, '\\', dir])
        if not os.path.isdir(new_test_dir):
            os.mkdir(new_test_dir)
        new_face_image = new_test_image.get_face_image()
        new_face_image.resize_image()
        new_path=''.join([test_dir, '\\', dir, '\\', new_test_image.file_name])
        old_path=new_test_image.path
        new_face_image.save(new_path)
        test_paths.append((new_path, old_path))

train_directories = [dir for dir in os.listdir(train_dir) if not '.' in dir]
for dir in train_directories:
    dir_path=''.join([train_dir, '\\', dir])
    files = os.listdir(dir_path)
    augmentation_images=[file for file in files if ((len(file.split('.'))==2) and (file.split('.')[1] in ['jpg', 'jpeg', 'png']) and file.split('_')[0]=='aug') ]

    for cur_image in augmentation_images:
        new_augmentation_image=image_in_set(''.join([dir_path, '\\', cur_image]))
        new_face_image=new_augmentation_image.get_face_image()
        os.remove(new_augmentation_image.path)
        if (new_face_image is None) or (isinstance(new_face_image, type(None))) or\
        (new_face_image.values.shape[0]<80) or (new_face_image.values.shape[1]<80) or (abs(new_face_image.values.shape[0]-new_face_image.values.shape[1])>80):
            continue
        else:
            new_face_image.resize_image()
            new_face_image.save(new_augmentation_image.path)
            augmentation_paths.append(new_face_image.path)

no_faces_detected_dir=''.join([dataset_dir, '\\', 'no faces detected'])
if not os.path.isdir(no_faces_detected_dir):
    os.mkdir(no_faces_detected_dir)
for image_path in no_faces_detected:
    no_face_image=image_in_set(image_path)
    no_face_image.save(''.join([no_faces_detected_dir, '\\', no_face_image.file_name]))


facenet_model = load_model('facenet_keras.h5',compile=False)

#all the images in the train, validation and test sets go through normalization
#the normalization type is standardization that is done by substracting the train set mean and dividing by the train set STD
#the normalized values are calculated by the normalize mothod
train_df=pd.DataFrame(columns=['id', 'name', 'embedding', 'path'])
for index,image_paths in enumerate(train_paths):
    cur_image = image_in_set(image_paths[0])
    cur_image_embedding=get_embedding(cur_image,"normalize_by_train_values",facenet_model,train_paths)
    train_df.loc[index]=[cur_image.id, cur_image.name, cur_image_embedding, image_paths[1]]

augmentation_df=pd.DataFrame(columns=['id', 'name', 'embedding'])
for index,image_path in enumerate(augmentation_paths):
    cur_image = image_in_set(image_path)
    cur_image_embedding=get_embedding(cur_image,"normalize_by_train_values",facenet_model,train_paths)
    augmentation_df.loc[index]=[cur_image.id, cur_image.name, cur_image_embedding]

validation_df=pd.DataFrame(columns=['id', 'name', 'embedding', 'path'])
for index,image_paths in enumerate(validation_paths):
    cur_image = image_in_set(image_paths[0])
    cur_image_embedding=get_embedding(cur_image,"normalize_by_train_values",facenet_model,train_paths)
    validation_df.loc[index]=[cur_image.id, cur_image.name, cur_image_embedding, image_paths[1]]

test_df=pd.DataFrame(columns=['id', 'name', 'embedding', 'path'])
for index,image_paths in enumerate(test_paths):
    cur_image = image_in_set(image_paths[0])
    cur_image_embedding=get_embedding(cur_image,"normalize_by_train_values",facenet_model,train_paths)
    test_df.loc[index]=[cur_image.id, cur_image.name, cur_image_embedding, image_paths[1]]

all_data_df = pd.concat([train_df,validation_df,test_df],ignore_index=True)
db_df=all_data_df.groupby(['id','name'],as_index=False).aggregate({'embedding':list, 'path':list})

train_df=pd.concat([train_df.drop('path',axis=1),augmentation_df],ignore_index=True)
validation_df.drop('path',axis=1, inplace=True)
test_df.drop('path',axis=1, inplace=True)

cvs_dir=''.join([dataset_dir,'\\csv_data'])
if not os.path.isdir(cvs_dir):
    os.mkdir(cvs_dir)
os.chdir(cvs_dir)
train_df.to_csv(''.join([os.getcwd(),'\\train.csv']),index=False)
validation_df.to_csv(''.join([os.getcwd(),'\\validation.csv']),index=False)
test_df.to_csv(''.join([os.getcwd(),'\\test.csv']),index=False)

os.chdir('../../')

