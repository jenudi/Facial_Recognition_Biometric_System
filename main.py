from images_classes import *
from augmentation import *
import pandas as pd
import os
import shutil
import math
import pickle

def main(directory_name=None):
    root_dir=os.getcwd()
    dataset_dir = os.path.join(root_dir, input("please enter the dataset directory name")) \
        if directory_name is None else os.path.join(root_dir, directory_name)
    directories = [dir for dir in os.listdir(dataset_dir) if not '.' in dir]
    train_df = pd.DataFrame(columns=['path','class','face_indexes','to_aug'])
    validation_df = pd.DataFrame(columns=['path','class','face_indexes'])
    train_df_index=0
    validation_df_index=0
    for dir in directories:
        dir_path = os.path.join(dataset_dir,dir)
        files = os.listdir(dir_path)
        images_paths = [os.path.join(dir_path,file) for file in files if\
                        ((len(file.split('.')) == 2) and (file.split('.')[1] in ['jpg', 'jpeg','png']))]  # contains all the images of the person in the current directory

        images_to_save_in_dfs=list()

        for image_path in images_paths:
            print(dir)
            if len(images_to_save_in_dfs)>=4:
                break
            image_object=ImageInSet(image_path)
            image_face_indexes=image_object.get_face_indexes()
            if (image_face_indexes is None) or (isinstance(image_face_indexes, type(None))):
                images_to_save_in_dfs.append((image_object,image_face_indexes))

        if len(images_to_save_in_dfs)==0:
            continue

        if len(images_to_save_in_dfs)==1:
            validation_df.loc[validation_df_index]=[images_to_save_in_dfs[0][0].path,images_to_save_in_dfs[0][0].id
                ,images_to_save_in_dfs[0][1]]
            validation_df_index+=1
            for i in range(50):
                print("50: i="+str(i))
                train_df_index.loc[train_df_index] = [images_to_save_in_dfs[0][0].path,
                                                     images_to_save_in_dfs[0][0].path.id, images_to_save_in_dfs[0][1], True]
                train_df_index+=1

        if len(images_to_save_in_dfs)==2:
            train_df_index.loc[train_df_index] = [images_to_save_in_dfs[0][0].path,
                                                  images_to_save_in_dfs[0][0].path.id, images_to_save_in_dfs[0][1], False]
            train_df_index += 1
            validation_df.loc[validation_df_index]=[images_to_save_in_dfs[1][0].path,images_to_save_in_dfs[1][0].id
                ,images_to_save_in_dfs[1][1]]
            validation_df_index+=1
            for i in range(49):
                print("49: i="+str(i))
                train_df_index.loc[train_df_index] = [images_to_save_in_dfs[0][0].path,
                                                     images_to_save_in_dfs[0][0].path.id, images_to_save_in_dfs[0][1], True]
                train_df_index+=1

        if len(images_to_save_in_dfs)==3:
            train_df_index.loc[train_df_index] = [images_to_save_in_dfs[0][0].path,
                                                  images_to_save_in_dfs[0][0].path.id, images_to_save_in_dfs[0][1], False]
            train_df_index += 1
            train_df_index.loc[train_df_index] = [images_to_save_in_dfs[1][0].path,
                                                  images_to_save_in_dfs[1][0].path.id, images_to_save_in_dfs[1][1], False]
            train_df_index += 1
            validation_df.loc[validation_df_index]=[images_to_save_in_dfs[2][0].path,images_to_save_in_dfs[2][0].id
                ,images_to_save_in_dfs[2][1]]
            validation_df_index+=1
            for i in range(24):
                print("24: i="+str(i))
                train_df_index.loc[train_df_index] = [images_to_save_in_dfs[0][0].path,
                                                     images_to_save_in_dfs[0][0].path.id, images_to_save_in_dfs[0][1], True]
                train_df_index+=1
                train_df_index.loc[train_df_index] = [images_to_save_in_dfs[1][0].path,
                                                     images_to_save_in_dfs[1][0].path.id, images_to_save_in_dfs[1][1], True]
                train_df_index+=1


    return train_df,validation_df

x,y=main("C:\\Users\Elad\\Desktop\\mini dataset")
print(x)
print(y)