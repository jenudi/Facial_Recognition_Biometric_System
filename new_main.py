from images_classes import *
import pandas as pd
import os
import shutil
import math
import pickle
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

root_dir=os.getcwd()
directory_name = 'data/dataset'
dataset_dir = os.path.join(root_dir, input("please enter the dataset directory path")) \
        if directory_name is None else os.path.join(root_dir, directory_name)
df = pd.DataFrame(columns=['path','name'])
directories = [dir for dir in os.listdir(dataset_dir) if not '.' in dir]
loop = tqdm(directories, position=0,leave=True)
for dir in loop:
    dir_path = os.path.join(dataset_dir, dir)
    files = os.listdir(dir_path)
    images_paths = [os.path.join(dir_path, file) for file in files if ((len(file.split('.')) == 2) and (file.split('.')[1] in ['jpg', 'jpeg', 'png']))]
    for img in images_paths:
        df = df.append({'path': img,'name':dir}, ignore_index=True)

df = df.groupby(by='name',as_index=False).agg({'path': lambda x: list(x)})
valid, indexes = list(), list()
loop = tqdm(df['path'], position=0,leave=True)
for index,paths in enumerate(loop):
    valid_pics, image_indexes = list(), list()
    for path in paths:
        if len(valid_pics) >= 3:
            break
        image_object = ImageInSet(path)
        image_face_indexes = image_object.get_face_indexes()
        if (not image_face_indexes is None) and (not isinstance(image_face_indexes, type(None))):
            valid_pics.append(path)
            image_indexes.append(image_face_indexes)
    valid.append(valid_pics)
    indexes.append(image_indexes)
df['valid'] = valid
df['indexes'] = indexes
index_to_drop = [index for index,value in enumerate(df['valid']) if len(value) == 0]
if len(index_to_drop) != 0:
    df.drop(index_to_drop,inplace=True)
    df.reset_index(drop=True,inplace=True)
df['cls'] = [i for i in range(len(df))]
df['num_of_valid'] = [len(i) for i in df['valid']]
dict_cls2name = dict(zip(df['cls'],df['name']))
id_to_name_dict = {value: key for key, value in dict_cls2name.items()}
#id_to_name_dict['Yonatan_Jenudi']

train_df = pd.DataFrame(columns=['path', 'class', 'face_indexes', 'aug','num_of_valid'])
validation_df = pd.DataFrame(columns=['path', 'class', 'face_indexes'])
loop = tqdm(df['num_of_valid'], position=0,leave=True)
for index,value in enumerate(loop):
    if value == 1:
        validation_df = validation_df.append({'path': df['valid'][index][0], 'class': df['cls'][index],'face_indexes': df['indexes'][index][0]}, ignore_index=True)
        train_df = train_df.append({'path': df['valid'][index][0], 'class': df['cls'][index],'face_indexes': df['indexes'][index][0],'aug':1,'num_of_valid':1}, ignore_index=True)
    if value == 2:
        validation_df = validation_df.append({'path': df['valid'][index][0], 'class': df['cls'][index],'face_indexes': df['indexes'][index][0]}, ignore_index=True)
        train_df = train_df.append({'path': df['valid'][index][1], 'class': df['cls'][index],'face_indexes': df['indexes'][index][1], 'aug': 0,'num_of_valid':0}, ignore_index=True)
        train_df = train_df.append({'path': df['valid'][index][1], 'class': df['cls'][index],'face_indexes': df['indexes'][index][1], 'aug': 1,'num_of_valid':2}, ignore_index=True)
    if value == 3:
        validation_df = validation_df.append({'path': df['valid'][index][0], 'class': df['cls'][index],'face_indexes': df['indexes'][index][0]}, ignore_index=True)
        train_df = train_df.append({'path': df['valid'][index][1], 'class': df['cls'][index],'face_indexes': df['indexes'][index][1], 'aug': 0,'num_of_valid':0}, ignore_index=True)
        train_df = train_df.append({'path': df['valid'][index][2], 'class': df['cls'][index],'face_indexes': df['indexes'][index][2], 'aug': 0,'num_of_valid':0}, ignore_index=True)
        train_df = train_df.append({'path': df['valid'][index][1], 'class': df['cls'][index],'face_indexes': df['indexes'][index][1], 'aug': 1,'num_of_valid':3}, ignore_index=True)
        train_df = train_df.append({'path': df['valid'][index][2], 'class': df['cls'][index],'face_indexes': df['indexes'][index][2], 'aug': 1,'num_of_valid':3}, ignore_index=True)

temp_list = list()
for i in train_df['num_of_valid']:
  if i == 0:
    temp_list.append(1/40)
  if i == 1:
    temp_list.append(39/40)
  if i == 2:
    temp_list.append(37/40)
  if i == 3:
    temp_list.append(18/40)
train_df['cls_w'] = temp_list
sampler = WeightedRandomSampler(train_df['cls_w'],num_samples=len(train_df['cls_w']),replacement=True)

#%%