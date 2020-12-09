import cv2 as cv
from google.colab.patches import cv2_imshow
import tensorflow as tf
import keras
import numpy as np
import os #enables getting file names from directory
from PIL import Image #we may also use skimage
from skimage import io #enables reading a single image
import random
import dlib
import os

#import face_recognition #run this only after installing dlib and face_recognition (above code)
#import importlib




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



def preprocessing_for_train(image):
    image=cv.resize(image, (256,256))
    image=cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image=add_filter(image)
    image=add_noise(image)
    return image

def preprocessing_for_val_and_test(image):
    image=cv.resize(image, (256,256))
    image=cv.cvtColor(image, cv.COLOR_BGR2RGB)
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
    preprocessing_function=preprocessing_for_train) #may also try nearest, constant, reflect, wrap. when using 'constant' we should add 'cval' value of 125



root_dir = '/content'
train_dir = root_dir + '/train'
os.chdir(train_dir)
directories = os.listdir(train_dir)
if os.path.exists(train_dir + '/.ipynb_checkpoints'):
    directories.remove('.ipynb_checkpoints')

images_sum = np.zeros((256, 256, 3))
num_of_train_images = 0
images = {}

for dir in directories:
    files = os.listdir(train_dir + '/' + dir)
    files_formats = [file.split('.') for file in files]
    images[dir] = ['.'.join(file_format) for file_format in files_formats if file_format[1] in ['jpg', 'jpeg', 'png']]

    if len(images[dir]) == 0:
        continue

    if len(images[dir]) in [1, 2, 3]:
        image = cv.imread(train_dir + '/' + dir + '/' + images[dir][0])
        image = cv.resize(image, (256, 256))
        images_sum += image
        num_of_train_images += 1

    else:
        for image_name in images[dir]:
            image = cv.imread(train_dir + '/' + dir + '/' + image_name)
            image = cv.resize(image, (256, 256))
            images_sum += image
            num_of_train_images += 1





root_dir = '/content'
train_dir = root_dir + '/train'
validation_dir = root_dir + '/validation'
test_dir = root_dir + '/test'
os.chdir(train_dir)

if not os.path.isdir(validation_dir):
    os.mkdir(validation_dir)
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)

directories = os.listdir(train_dir)
if os.path.exists(train_dir + '/.ipynb_checkpoints'):
    directories.remove('.ipynb_checkpoints')

for dir in directories:
    files = os.listdir(train_dir + '/' + dir)
    files_formats = [file.split('.') for file in files]
    images = ['.'.join(file_format) for file_format in files_formats if file_format[1] in ['jpg', 'jpeg', 'png']]

    if len(images) == 0:
        continue

    for image_name in images:

        image = cv.imread(train_dir + '/' + dir + '/' + image_name)

        if (len(images) == 1) or (len(images) == 2 and os.path.isdir(validation_dir + '/' + dir)) or (
        os.path.isdir(test_dir + '/' + dir)):

            image = image.reshape((1,) + image.shape)
            i, j = 0, 0
            for batch in datagen.flow(image,
                                      batch_size=5,
                                      save_to_dir=train_dir + '/' + dir,
                                      save_prefix='aug',
                                      save_format='jpg'):
                i += 1
                if i == 5:
                    break

        elif len(images) > 1:
            image = preprocessing_for_val_and_test(image)
            os.remove(train_dir + '/' + dir + '/' + image_name)

            if not os.path.isdir(validation_dir + '/' + dir):
                os.mkdir(validation_dir + '/' + dir)
                cv.imwrite(validation_dir + '/' + dir + '/' + 'val_' + image_name, image)

            elif not os.path.isdir(test_dir + '/' + dir):
                os.mkdir(test_dir + '/' + dir)
                cv.imwrite(test_dir + '/' + dir + '/' + 'test_' + image_name, image)