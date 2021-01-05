import cv2 as cv
import numpy as np
import random
from keras.preprocessing.image import ImageDataGenerator
from skimage.util import random_noise

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
    preprocessing_function=preprocessing_for_augmentation
)