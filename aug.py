import cv2 as cv
import numpy as np
import imgaug.augmenters as iaa


def aug_img(img):
    img = np.expand_dims(img, axis=0)
    one = iaa.OneOf([iaa.GaussianBlur(sigma=(0.0, 3.0)),
                     iaa.AverageBlur(k=(2, 11)),
                     iaa.AverageBlur(k=((5, 11), (1, 3))),
                     iaa.MedianBlur(k=(3, 11)),
                     iaa.MeanShiftBlur(),])
    two = iaa.OneOf([iaa.WithBrightnessChannels(iaa.Add((-50, 50))),
                     iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
                     iaa.MultiplyBrightness((0.5, 1.5)),
                     iaa.AddToBrightness((-30, 30)),
                     iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
                     iaa.RemoveSaturation()])
    three = iaa.OneOf([iaa.Affine(scale=(0.5, 1.5)),
                       iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
                       iaa.Affine(rotate=(-45, 45)),
                       iaa.Affine(shear=(-16, 16)),
                       iaa.PerspectiveTransform(scale=(0.01, 0.15)),])
    simetimes2 = iaa.Sometimes(0.25, two)
    simetimes3 = iaa.Sometimes(0.2,one)
    seq = iaa.Sequential([three,simetimes2,simetimes3],random_order=True,)
    seq2 = iaa.Sequential([seq,iaa.Resize(160)],random_order=False)
    images_aug = seq2(images=img)
    return images_aug[0]


def aug_img2(img):
    img = np.expand_dims(img, axis=0)
    seq2 = iaa.Sequential([iaa.Resize(160)], random_order=False)
    images_aug = seq2(images=img)
    return images_aug[0]