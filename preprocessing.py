import cv2 as cv
import numpy as np
import os #enables getting file names from directory
import random
import math
import sys
from keras.preprocessing.image import ImageDataGenerator
from face_recognition import face_locations #run this only after installing dlib, cmake and face_recognition
from skimage.util import random_noise


#classes for train set images, validation set images and test set images
class image_in_set:

    def __init__(self,path):
        self.values=cv.imread(path)
        self.path=path
        self.dir=self.path.split('\\')[-2]
        self.person=' '.join(self.dir.split('_'))
        self.file_name=self.path.split('\\')[-1]

    def save(self,new_path):
        if not type(self).__name__=='image_in_set':
            class_paths_list=getattr(sys.modules[__name__], type(self).__name__).paths_list
            getattr(sys.modules[__name__], type(self).__name__).paths_list[class_paths_list.index(self.path)] = new_path
        self.path=new_path
        cv.imwrite(self.path,self.values)

    def detect_face(self):
        face_loc=face_locations(self.values)
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
class train_image(image_in_set):

    paths_list=[]

    def __init__(self,path):
        image_in_set.__init__(self,path)

        train_image.paths_list.append(path)

    @classmethod
    def get_train_mean(cls):
        if len(cls.paths_list)>0:
            train_mean=[]
            for train_path in cls.paths_list:
                train_image=image_in_set(train_path)
                train_mean.append(train_image.values)
                return np.mean(train_mean,axis=(0,1,2))
        else:
            raise ValueError("No train images")

    @classmethod
    def get_train_std(cls):
        if len(cls.paths_list)>0:
            train_std=[]
            for train_path in cls.paths_list:
                train_image=image_in_set(train_path)
                train_std.append(train_image.values)
                return np.std(train_std,axis=(0,1,2))
        else:
            raise ValueError("No train images")

class augmentation_image(image_in_set):

    paths_list=[]

    def __init__(self,path):
        image_in_set.__init__(self,path)

        augmentation_image.paths_list.append(path)


class validation_image(image_in_set):

    paths_list=[]

    def __init__(self,path):
        image_in_set.__init__(self,path)

        validation_image.paths_list.append(path)


class test_image(image_in_set):

    paths_list=[]

    def __init__(self,path):
        image_in_set.__init__(self,path)

        test_image.paths_list.append(path)



#filters and noise are randomly added to the augmentation images
def identity_filter(img):
    filtered_image = img.copy()
    kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    return cv.filter2D(filtered_image, -1, kernel).astype(np.uint8)

def averageing_filter(img):
    filtered_image = img.copy()
    return cv.blur(filtered_image,(5,5)).astype(np.uint8)

def gaussian_filter(img):
    filtered_image = img.copy()
    return cv.GaussianBlur(filtered_image,(5,5),2).astype(np.uint8)

def median_filter(img):
    filtered_image = img.copy()
    return cv.medianBlur(filtered_image,5).astype(np.uint8)

def bileteral_filter(img):
    filtered_image = img.copy()
    return cv.bilateralFilter(filtered_image,5,125,100).astype(np.uint8)

def laplacian_filter(img):
    filtered_image = img.copy()
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv.filter2D(filtered_image, -1, kernel).astype(np.uint8)

def add_filter(img):
  filters ={0:identity_filter, 1:identity_filter, 2:averageing_filter,
            3:gaussian_filter, 4:median_filter,
            5:bileteral_filter, 6:laplacian_filter}
  return filters[random.randint(0,6)](img)

def salt_and_paper_noise(img):
    noisy_image=img.copy()
    rnd = np.random.rand(img.shape[0], img.shape[1])
    noisy_image[rnd < 0.02] = 255
    noisy_image[rnd > 0.98] = 0
    return noisy_image.astype(np.uint8)

def gaussian_noise(img):
    noisy_image=img.copy()
    return (255*random_noise(noisy_image, mode='gaussian', var=0.05 ** 2)).astype(np.uint8)

def poisson_noise(img):
    noisy_image=img.copy()
    return (255*random_noise(noisy_image, mode="poisson")).astype(np.uint8)

def speckle_noise(img):
    noisy_image=img.copy()
    return (255*random_noise(noisy_image, mode="speckle")).astype(np.uint8)

def add_noise(img):
   noises={0:identity_filter, 1:salt_and_paper_noise, 2:gaussian_noise,
            3:poisson_noise, 4:speckle_noise}
   return noises[random.randint(0,4)](img)


def preprocessing_for_augmentation(img):
    RGB_img=cv.cvtColor(img, cv.COLOR_BGR2RGB)
    filtered_image=add_filter(RGB_img)
    noisy_image=add_noise(filtered_image)
    return noisy_image



datagen = ImageDataGenerator(
    brightness_range=(0.5, 1.5),
    zoom_range = 0.2,
    shear_range = 0.2,
    rotation_range = 30, #Random rotation between 0-30 degrees
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'reflect', #may also try nearest, constant, reflect, wrap. when using 'constant' we should add 'cval' value of 125
    preprocessing_function=preprocessing_for_augmentation)


def main():
    dataset_dir=input("Please enter the dataset directory path")
    os.chdir(dataset_dir)
    directories = [dir for dir in os.listdir(dataset_dir) if not '.' in dir] #directories contain all the people that have images in the dataset

    #new direstories are being made for the train, validation and test sets
    train_dir = ''.join([dataset_dir, '\\train'])
    validation_dir = ''.join([dataset_dir, '\\validation'])
    test_dir = ''.join([dataset_dir, '\\test'])
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    if not os.path.isdir(validation_dir):
        os.mkdir(validation_dir)
    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)

    no_faces_detected=[]

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
            face=cur_image.detect_face()
            if face is None:
                images.remove(image_name)
                print(' '.join(["No face detected for",cur_image.person]))
                no_faces_detected.append(cur_image.path)

        cur_train_set=[]
        cur_test_set = []
        cur_validation_set = []
        if len(images)==2:
            cur_train_set.append(train_image(''.join([dir_path, '\\', images[0]])))
            cur_test_set.append(test_image(''.join([dir_path, '\\', images[1]])))
        elif len(images)==3:
            cur_train_set.append(train_image(''.join([dir_path, '\\', images[0]])))
            cur_validation_set.append(validation_image(''.join([dir_path, '\\', images[1]])))
            cur_test_set.append(test_image(''.join([dir_path, '\\', images[2]])))
        elif len(images)>3:
            cur_train_set=[train_image(''.join([dir_path, '\\', cur_image])) for cur_image in images[0:math.ceil(len(images)*0.5)]]
            cur_validation_set=[validation_image(''.join([dir_path, '\\', cur_image])) for cur_image in images[math.ceil(len(images)*0.5):round(len(images)*0.75)]]
            cur_test_set=[test_image(''.join([dir_path, '\\', cur_image])) for cur_image in images[round(len(images) * 0.75):]]

    #every image that goes to the train set generates 5 new augmentad images
    #the image face locations are saved by the method detect_face
        for new_train_image in cur_train_set:
            new_train_dir=''.join([train_dir, '\\', dir])
            if not os.path.isdir(new_train_dir):
                os.mkdir(new_train_dir)
            new_train_image.preprocess()
            new_path=''.join([train_dir, '\\', dir, '\\', new_train_image.file_name])
            new_train_image.save(new_path)

            image_for_aug = new_train_image.values.reshape((1,) + new_train_image.values.shape)
            i=0
            for batch in datagen.flow(image_for_aug, batch_size=1, save_to_dir=new_train_dir, save_prefix='aug', save_format='jpg'):
                i += 1
                if i == 5:
                    break

        for new_validation_image in cur_validation_set:
            new_validation_dir=''.join([validation_dir, '\\', dir])
            if not os.path.isdir(new_validation_dir):
                os.mkdir(new_validation_dir)
            new_validation_image.preprocess()
            new_path=''.join([validation_dir, '\\', dir, '\\', new_validation_image.file_name])
            new_validation_image.save(new_path)

        for new_test_image in cur_test_set:
            new_test_dir=''.join([test_dir, '\\', dir])
            if not os.path.isdir(new_test_dir):
                os.mkdir(new_test_dir)
            new_path = ''.join([test_dir, '\\', dir, '\\', new_test_image.file_name])
            new_test_image.preprocess()
            new_test_image.save(new_path)

    train_directories = [dir for dir in os.listdir(train_dir) if not '.' in dir]
    for dir in train_directories:
        dir_path=''.join([train_dir, '\\', dir])
        files = os.listdir(dir_path)
        augmentation_images=[file for file in files if ((len(file.split('.'))==2) and (file.split('.')[1] in ['jpg', 'jpeg', 'png']) and file.split('_')[0]=='aug') ]

        for cur_image in augmentation_images:
            new_augmentation_image=augmentation_image(''.join([dir_path, '\\', cur_image]))
            face=new_augmentation_image.detect_face()
            if face is None:
                augmentation_image.paths_list.remove(new_augmentation_image.path)
                os.remove(new_augmentation_image.path)

    no_faces_detected_dir=''.join([dataset_dir, '\\', 'no faces detected'])
    if not os.path.isdir(no_faces_detected_dir):
        os.mkdir(no_faces_detected_dir)
    for image_path in no_faces_detected:
        no_face_image=image_in_set(image_path)
        no_face_image.save(''.join([no_faces_detected_dir, '\\', no_face_image.file_name]))

main()
train_set=sum([train_image.paths_list,augmentation_image.paths_list],[])
validation_set=validation_image.paths_list
test_set=test_image.paths_list