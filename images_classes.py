from face_recognition import face_locations
import cv2 as cv
import numpy as np


class Image_in_set:

    name_to_id_dict=dict()

    def __init__(self,path):
        self.values=cv.imread(path)
        self.path=path
        self.dir=self.path.split('\\')[-2]
        self.file_name=self.path.split('\\')[-1]
        self.name=' '.join(self.dir.split('_'))
        self.id=Image_in_set.name_to_id(self.name)

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
            return Face_image(face,self.name)
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

    def get_embedding(self, normalization_method, model, train_paths_list=None):
        if normalization_method == "normalize_by_train_values":
            assert (not train_paths_list is None) or (not isinstance(train_paths_list, type(None))), "enter train paths list in order to use the normalize by train values method"
            norm_values = self.normalize_by_train_values(train_paths_list).astype("float32")
        else:
            norm_values = self.normalize_by_image_values()
        four_dim_values = np.expand_dims(norm_values, axis=0)
        embedding = model.predict(four_dim_values)[0]
        return embedding


class Captured_frame(Image_in_set):

    def __init__(self,values):
        self.values=values
        self.name=None
        self.path=None
        self.face_image=self.get_face_image
        self.face_detected=True if (self.face_image is not None) and not (isinstance(self.face_image, type(None))) else False

    def set_name(self,name):
        self.name=name
        if ((self.face_image is not None) and not (isinstance(self.face_image, type(None)))):
            self.face_image.name=name


class Face_image(Image_in_set):

    def __init__(self,values,name):
        self.values=values
        self.name=name
        self.path=None


def get_images_mean(paths_list):
    assert len(paths_list), "paths list must not be empty"
    train_mean=list()
    for path in paths_list:
        train_image=Image_in_set(path[0])
        train_mean.append(train_image.values)
    return np.mean(train_mean, axis=(0, 1, 2))

def get_images_std(paths_list):
    assert len(paths_list), "paths list must not be empty"
    train_std=list()
    for path in paths_list:
        train_image=Image_in_set(path[0])
        train_std.append(train_image.values)
    return np.std(train_std,axis=(0,1,2))