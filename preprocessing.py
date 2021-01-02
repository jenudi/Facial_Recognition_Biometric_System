import cv2 as cv
import keras
import numpy as np
import os #enables getting file names from directory
import random
import face_recognition #run this only after installing dlib, cmake and face_recognition
import math

#classes for train set images, validation set images and test set images
class image:

    def __init__(self,path):
        self.values=cv.imread(path)
        self.path=path
        self.dir=self.path.split('/')[-2]
        self.person=' '.join(self.dir.split('_'))
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


if __name__ == "__main__":

    dataset_dir = 'C:/Users/gash5/Desktop/dataset'
    os.chdir(dataset_dir)
    directories = [dir for dir in os.listdir(dataset_dir) if not '.' in dir] #directories contain all the people that have images in the dataset

    #new direstories are being made for the train, validation and test sets
    train_dir = ''.join([dataset_dir, '/train'])
    validation_dir = ''.join([dataset_dir, '/validation'])
    test_dir = ''.join([dataset_dir, '/test'])
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    if not os.path.isdir(validation_dir):
        os.mkdir(validation_dir)
    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)

    no_faces_detected=[]

    for dir in directories:
        dir_path=''.join([dataset_dir, '/', dir])
        files = os.listdir(dir_path)
        images=[file for file in files if ((len(file.split('.')) == 2) and (file.split('.')[1] in ['jpg', 'jpeg', 'png']))] #contains all the images of the person in the current directory
        if len(images)<2:
            continue

        #if a person has less than 3 images he will not be part of any set
        #if a person has 3 images all the images will be put in the train set
        #if a person has 4 images all the images except one will he put in the train set and the last one will be put on the validation set
        #if a person has more than 4 images  all the images except two will he put in the train set and the last two will be put on the validation set and test set
        for image_name in images:
            cur_image=image(''.join([dir_path, '/', image_name]))
            face=cur_image.detect_face()
            if face is None:
                images.remove(image_name)
                print(' '.join(["No face detected for",cur_image.person]))
                no_faces_detected.append(cur_image.path)

        cur_train_set=[]
        cur_test_set = []
        cur_validation_set = []
        if len(images)==2:
            cur_train_set.append(train_image(''.join([dir_path, '/', images[0]])))
            cur_test_set.append(test_image(''.join([dir_path, '/', images[1]])))
        elif len(images)==3:
            cur_train_set.append(train_image(''.join([dir_path, '/', images[0]])))
            cur_test_set.append(test_image(''.join([dir_path, '/', images[1]])))
            cur_validation_set.append(validation_image(''.join([dir_path, '/', images[2]])))
        elif len(images)>3:
            cur_train_set=[train_image(''.join([dir_path, '/', cur_image])) for cur_image in images[0:math.floor(len(images)*0.5)]]
            cur_test_set=[test_image(''.join([dir_path, '/', cur_image])) for cur_image in images[math.floor(len(images)*0.5):math.floor(len(images)*0.75)]]
            cur_validation_set=[validation_image(''.join([dir_path, '/', cur_image])) for cur_image in images[math.floor(len(images)*0.75):]]

    #every image that goes to the train set generates 5 new augmentad images
    #the image face locations are saved by the method detect_face
        for new_train_image in cur_train_set:
            new_train_dir=''.join([train_dir, '/', dir])
            if not os.path.isdir(new_train_dir):
                os.mkdir(new_train_dir)
            new_train_image.preprocess()
            new_path=''.join([train_dir, '/', dir, '/', image_name])
            train_image.train_paths_list[train_image.train_paths_list.index(new_train_image.path)]=new_path
            new_train_image.save(new_path)

            image_for_aug = new_train_image.values.reshape((1,) + new_train_image.values.shape)
            i=0
            for batch in datagen.flow(image_for_aug, batch_size=1, save_to_dir=new_train_dir, save_prefix='aug', save_format='jpg'):
                i += 1
                if i == 5:
                    break

        for new_test_image in cur_test_set:
            new_test_dir=''.join([test_dir, '/', dir])
            if not os.path.isdir(new_test_dir):
                os.mkdir(new_test_dir)
            new_path = ''.join([test_dir, '/', dir, '/', image_name])
            new_test_image.preprocess()
            test_image.test_paths_list[test_image.test_paths_list.index(new_test_image.path)] = new_path
            new_test_image.save(new_path)


        for new_validation_image in cur_validation_set:
            new_validation_dir=''.join([validation_dir, '/', dir])
            if not os.path.isdir(new_validation_dir):
                os.mkdir(new_validation_dir)
            new_validation_image.preprocess()
            new_path=''.join([validation_dir, '/', dir, '/', image_name])
            validation_image.validation_paths_list[validation_image.validation_paths_list.index(new_validation_image.path)]=new_path
            new_validation_image.save(new_path)

    train_directories = [dir for dir in os.listdir(train_dir) if not '.' in dir]
    for dir in train_directories:
        dir_path=''.join([train_dir, '/', dir])
        files = os.listdir(dir_path)
        augmentation_images=[file for file in files if ((len(file.split('.'))==2) and (file.split('.')[1] in ['jpg', 'jpeg', 'png']) and file.split('_')[0]=='aug') ]

        for cur_image in augmentation_images:
            new_augmentation_image=augmentation_image(''.join([dir_path, '/', cur_image]))
            face=new_augmentation_image.detect_face()
            if face is None:
                augmentation_image.augmentation_paths_list.remove(new_augmentation_image.path)
                os.remove(new_augmentation_image.path)

    no_faces_detected_dir=''.join([dataset_dir, '/', 'no faces detected'])
    if not os.path.isdir(no_faces_detected_dir):
        os.mkdir(no_faces_detected_dir)
    for image_path in no_faces_detected:
        no_face_image=image(image_path)
        no_face_image.save(''.join([no_faces_detected_dir, '/', no_face_image.file_name]))

    train_set=sum([train_image.train_paths_list,augmentation_image.augmentation_paths_list],[])
    validation_set=validation_image.validation_paths_list
    test_set=train_image.train_paths_list
