import cv2 as cv
import numpy as np
import random
from keras.preprocessing.image import ImageDataGenerator
from skimage.util import random_noise
import imgaug.augmenters as iaa

#filters and noise are randomly added to the augmentation images

def identity_filter(img):
    filtered_image = img.copy()
    kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    return cv.filter2D(filtered_image, -1, kernel).astype(np.uint8)

def averageing_filter(img): #blurs image by averaging surrounding pixel values
    filtered_image = img.copy()
    return cv.blur(filtered_image,(5,5)).astype(np.uint8)

def gaussian_filter(img): #deals with gaussian noise, due to sensor or electronic noise
    filtered_image = img.copy()
    return cv.GaussianBlur(filtered_image,(5,5),2).astype(np.uint8)

def median_filter(img): #replaces value by an EXISTING value in the surrounding pixels. works well for S&P noise
    filtered_image = img.copy()
    return cv.medianBlur(filtered_image,5).astype(np.uint8)

def bileteral_filter(img): #blurs pixles by taking into account nearby pixels of similar intensity, therby preserving edges
    filtered_image = img.copy()
    return cv.bilateralFilter(filtered_image,5,125,100).astype(np.uint8) #(src, distance, sigmaColor, SigmaSpace)

def add_filter(img): #randomly assigns a filter to image
    filters ={0:identity_filter, 1:identity_filter, 2:averageing_filter,
              3:gaussian_filter, 4:median_filter,
              5:bileteral_filter}
    return filters[random.randint(0,5)](img)

def salt_and_paper_noise(img): #colors pixels white or black based on random values
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

def add_noise(img): #randomly assigns noise to image
    noises={0:identity_filter, 1:salt_and_paper_noise, 2:gaussian_noise,
            3:poisson_noise, 4:speckle_noise}
    return noises[random.randint(0,4)](img)

def preprocessing_for_augmentation(img):
    RGB_img=cv.cvtColor(img, cv.COLOR_BGR2RGB)
    filtered_image=add_filter(RGB_img)
    noisy_image=add_noise(filtered_image)
    return noisy_image #why return only noisy image?


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


def aug_img(img):
    img = np.expand_dims(img, axis=0)
    one = iaa.OneOf([iaa.Affine(scale=(0.9,1.1),mode='constant'),
                     iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},mode='constant'),
                     iaa.Affine(rotate=(-5, 5),mode='constant'),
                     iaa.Affine(shear=(-6, 6),mode='constant'),
                     iaa.ScaleX((0.9, 1.1)),
                     iaa.ScaleY((0.9, 1.1)),
                     iaa.PerspectiveTransform(scale=(0.01, 0.05))])
    two = iaa.OneOf([iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)),
                     iaa.AdditiveLaplaceNoise(scale=(0, 0.1 * 255)),
                     #iaa.Cutout(nb_iterations=1,size=0.1, squared=False,fill_mode="gaussian"),
                     #iaa.CoarseDropout(0.01, size_percent=0.9),
                    iaa.Salt(0.05)])
    three = iaa.OneOf([iaa.GaussianBlur(sigma=1.0),
                        iaa.imgcorruptlike.Fog(severity=1),
                        iaa.imgcorruptlike.Spatter(severity=1)])
    simetimes2 = iaa.Sometimes(0.05, two)
    simetimes3 = iaa.Sometimes(0.05,three)
    seq = iaa.Sequential([one,simetimes2,simetimes3],random_order=True)
    images_aug = seq(images=img)
    return images_aug[0]
