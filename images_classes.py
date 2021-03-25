import cv2 as cv
import numpy as np
from random import randint,uniform
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch import from_numpy
from pymongo import MongoClient
#from face_recognition import face_locations
from PIL import Image, ImageFile


class ImageInSet:

    mtcnn = MTCNN(post_process=False, image_size=160)
    face_detection_threshold=0.75
    face_recognition_threshold = 0.95
    face_recognition_model = InceptionResnetV1(pretrained='vggface2').eval()
    name_to_id_dict=dict()
    image_size=(160,160)

    def __init__(self,path):
        #self.values=cv.imread(path)
        self.values = Image.open(path)
        self.path=path
        self.dir=self.path.split('\\')[-2]
        self.file_name=self.path.split('\\')[-1]
        self.name=' '.join(self.dir.split('_'))
        self.id=ImageInSet.name_to_id(self.name)

    @classmethod
    def name_to_id(cls,name):
        if name in cls.name_to_id_dict.keys():
            return cls.name_to_id_dict[name]
        else:
            new_id=len(cls.name_to_id_dict.keys())
            cls.name_to_id_dict[name]=new_id
            return new_id

    '''
    def get_face_image(self):
        face_loc=face_locations(self.values)
        try:
            face=self.values[face_loc[0][0]:face_loc[0][1],face_loc[0][3]:face_loc[0][2]]
        except IndexError:
            return None
        if (not face is None) and (not isinstance(face, type(None))) and len(face):
            return Face_image(face,self.name)
        else:
            return None
    '''

    def get_face_indexes(self):
        boxes, probs = ImageInSet.mtcnn.detect(self.values, landmarks=False)
        if (not boxes is None) and (not isinstance(boxes, type(None))):
            if probs[0]>= ImageInSet.face_detection_threshold:
                #face_indexes=[int(b) for b in boxes[0]]
                return boxes[0]
        else:
            return None

    def get_face_image(self,indexes_box):
        pil_image=Image.open(self.path)
        pil_image.crop(indexes_box)
        return FaceImage(pil_image)
        #return FaceImage(self.values[int(indexes[1]):int(indexes[3]), int(indexes[0]):int(indexes[2])], self.name)

    def normalize_by_train_values(self,train_mean,train_std):
        return (self.values - train_mean) / train_std

    def normalize_by_image_values(self):
        return (self.values- self.values.mean())/self.values.std()

    def save(self, new_path):
        self.path = new_path
        cv.imwrite(self.path, self.values)

    def resize_image(self):
        self.values = cv.resize(self.values, ImageInSet.image_size)

    def get_embedding(self, normalization_method, train_mean=None, train_std=None):
        if normalization_method == "normalize_by_train_values":
            assert (not train_mean is None) and (not train_std is None), "enter train paths list in order to use the normalize by train values method"
            norm_values = self.normalize_by_train_values(train_mean,train_std).astype("float32")
        else:
            norm_values = self.normalize_by_image_values()
        #four_dim_values = np.expand_dims(norm_values, axis=0)
        #embedding = model.predict(four_dim_values)[0]
        #tensor_norm_values=from_numpy(norm_values).unsqueeze(0)
        norm_values_axis_swapped_0_2=np.swapaxes(norm_values,0,2)
        norm_values_axis_swapped_1_2=np.swapaxes(norm_values_axis_swapped_0_2,1,2)
        tensor_norm_values=from_numpy(norm_values_axis_swapped_1_2).unsqueeze(0)
        embedding = ImageInSet.face_recognition_model(tensor_norm_values)
        return embedding.detach()[0]


class FaceImage:

    def __init__(self,values,name):
        self.values=values
        self.name=name
        self.path=None

    def save(self,path):
        self.values.save(path)
        self.path=path


def get_images_mean(paths_list):
    assert len(paths_list), "paths list must not be empty"
    train_mean=list()
    for path in paths_list:
        train_image=ImageInSet(path[0])
        train_mean.append(train_image.values)
    return np.mean(train_mean, axis=(0, 1, 2))

def get_images_std(paths_list):
    assert len(paths_list), "paths list must not be empty"
    train_std=list()
    for path in paths_list:
        train_image=ImageInSet(path[0])
        train_std.append(train_image.values)
    return np.std(train_std,axis=(0,1,2))