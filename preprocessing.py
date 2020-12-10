import cv2 as cv
import keras
import numpy as np
import os #enables getting file names from directory
import random
#from PIL import Image #we may also use skimage
#from skimage import io #enables reading a single image
#import dlib
#import face_recognition #run this only after installing dlib and face_recognition (above code)
#import tensorflow as tf





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
   noises={0:identity_filter, 1:salt_and_paper_noise, 2:gaussian_noise,
            3:poisson_noise}
   return noises[random.randint(0,3)](image)

def preprocessing_for_augmantation(image):
    image=cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image=add_filter(image)
    image=add_noise(image)
    return image

def preprocessing_for_train_val_and_test(image):
    image=cv.resize(image, (256,256))
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
    fill_mode = 'reflect',
    preprocessing_function=preprocessing_for_augmantation) #may also try nearest, constant, reflect, wrap. when using 'constant' we should add 'cval' value of 125




dataset_dir = 'C:/Users/gash5/Desktop/dataset'
os.chdir(dataset_dir)
directories = [dir for dir in os.listdir(dataset_dir) if not '.' in dir]

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
    files_formats = [file.split('.') for file in files if len(file.split('.'))==2]
    images = ['.'.join(file_format) for file_format in files_formats if file_format[1] in ['jpg', 'jpeg', 'png']]

    if len(images)<3:
        continue

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

    for image_name in train_set:
        image = cv.imread(dataset_dir + '/' + dir + '/' + image_name)
        image_save=preprocessing_for_train_val_and_test(image)
        cv.imwrite(train_dir + '/' + dir + '/' + image_name, image_save)

        image_for_aug = image_save.reshape((1,) + image_save.shape)
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
        image = cv.imread(dataset_dir + '/' + dir + '/' + image_name)
        image_save = preprocessing_for_train_val_and_test(image)
        cv.imwrite(validation_dir + '/' + dir + '/' + image_name, image_save)

    for image_name in test_set:
        if not os.path.isdir(test_dir + '/' + dir):
            os.mkdir(test_dir + '/' + dir)
        image = cv.imread(dataset_dir + '/' + dir + '/' + image_name)
        image_save = preprocessing_for_train_val_and_test(image)
        cv.imwrite(test_dir + '/' + dir + '/' + image_name, image_save)




os.chdir(train_dir)
train_directories = [dir for dir in os.listdir(train_dir) if not '.' in dir]

train_images=[]
number_of_train_images=0

for dir in train_directories:
    files = os.listdir(train_dir + '/' + dir)
    files_formats = [file.split('.') for file in files if len(file.split('.'))==2]
    images = ['.'.join(file_format) for file_format in files_formats if file_format[1] in ['jpg', 'jpeg', 'png']]
    for image_name in images:
        image = cv.imread(train_dir + '/' + dir + '/' + image_name)
        train_images.append(cv.resize(image, (256,256)))
        number_of_train_images+=1

train_images_mean=np.mean(train_images,axis=(0,1,2))
train_images_std=np.std(train_images,axis=(0,1,2))


validation_directories = [dir for dir in os.listdir(validation_dir) if not '.' in dir]
test_directories = [dir for dir in os.listdir(test_dir) if not '.' in dir]

for dir in train_directories:
    if not os.path.isdir(train_dir + '/' + dir + '/' + 'images_for_cnn'):
        os.mkdir(train_dir + '/' + dir + '/' + 'images_for_cnn')
    files = os.listdir(train_dir + '/' + dir)
    files_formats = [file.split('.') for file in files if len(file.split('.'))==2]
    images = ['.'.join(file_format) for file_format in files_formats if file_format[1] in ['jpg', 'jpeg', 'png']]

    for image_name in images:
        image = cv.imread(train_dir + '/' + dir + '/' + image_name)
        image_save = (image-train_images_mean)/train_images_std
        cv.imwrite(train_dir + '/' + dir + '/' + 'images_for_cnn'  + '/' + image_name , image_save)

for dir in validation_directories:
    if not os.path.isdir(validation_dir + '/' + dir + '/' + 'images_for_cnn'):
        os.mkdir(validation_dir + '/' + dir + '/' + 'images_for_cnn')
    files = os.listdir(validation_dir + '/' + dir)
    files_formats = [file.split('.') for file in files if len(file.split('.'))==2]
    images = ['.'.join(file_format) for file_format in files_formats if file_format[1] in ['jpg', 'jpeg', 'png']]

    for image_name in images:
        image = cv.imread(validation_dir + '/' + dir + '/' + image_name)
        image_save = (image-train_images_mean)/train_images_std
        cv.imwrite(validation_dir + '/' + dir + '/' + 'images_for_cnn' + '/' + image_name, image_save)

for dir in test_directories:
    if not os.path.isdir(test_dir + '/' + dir + '/' + 'images_for_cnn'):
        os.mkdir(test_dir + '/' + dir + '/' + 'images_for_cnn')
    files = os.listdir(test_dir + '/' + dir)
    files_formats = [file.split('.') for file in files if len(file.split('.'))==2]
    images = ['.'.join(file_format) for file_format in files_formats if file_format[1] in ['jpg', 'jpeg', 'png']]

    for image_name in images:
        image = cv.imread(test_dir + '/' + dir + '/' + image_name)
        image_save = (image-train_images_mean)/train_images_std
        cv.imwrite(test_dir + '/' + dir + '/' + 'images_for_cnn' + '/' + image_name, image_save)