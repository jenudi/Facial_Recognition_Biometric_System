from images_classes import *
import pandas as pd
import os
import shutil
import math
import pickle


if __name__=="__main__":

    root_dir=os.getcwd()
    directory_name = None
    dataset_dir = os.path.join(root_dir, input("please enter the dataset directory path")) \
        if directory_name is None else os.path.join(root_dir, directory_name)
    directories = [dir for dir in os.listdir(dataset_dir) if not '.' in dir]
    train_df = pd.DataFrame(columns=['path','class','face_indexes','to_aug'])
    validation_df = pd.DataFrame(columns=['path','class','face_indexes'])
    images_to_save_in_dfs = list()

    for dir in directories:
        images_to_save_in_dfs.clear()
        dir_path = os.path.join(dataset_dir,dir)
        files = os.listdir(dir_path)
        images_paths = [os.path.join(dir_path,file) for file in files if\
                        ((len(file.split('.')) == 2) and (file.split('.')[1] in ['jpg', 'jpeg','png']))]  # contains all the images of the person in the current directory

        for image_path in images_paths:
            if len(images_to_save_in_dfs)>=3:
                break
            image_object=ImageInSet(image_path)
            image_face_indexes=image_object.get_face_indexes()
            if (not image_face_indexes is None) and (not isinstance(image_face_indexes, type(None))):
                images_to_save_in_dfs.append((image_object,image_face_indexes))

        if len(images_to_save_in_dfs)==0:
            del ImageInSet.name_to_id_dict[image_object.name]

        if len(images_to_save_in_dfs)==1:
            validation_df.loc[validation_df.shape[0]]=[images_to_save_in_dfs[0][0].path,images_to_save_in_dfs[0][0].id,
                images_to_save_in_dfs[0][1]]
            for i in range(50):
                train_df.loc[train_df.shape[0]] = [images_to_save_in_dfs[0][0].path,
                                                     images_to_save_in_dfs[0][0].id, images_to_save_in_dfs[0][1], True]

        if len(images_to_save_in_dfs)==2:
            train_df.loc[train_df.shape[0]] = [images_to_save_in_dfs[0][0].path,
                                                  images_to_save_in_dfs[0][0].id, images_to_save_in_dfs[0][1], False]
            validation_df.loc[validation_df.shape[0]] = [images_to_save_in_dfs[1][0].path,images_to_save_in_dfs[1][0].id
                ,images_to_save_in_dfs[1][1]]
            for i in range(49):
                train_df.loc[train_df.shape[0]] = [images_to_save_in_dfs[0][0].path,
                                                     images_to_save_in_dfs[0][0].id, images_to_save_in_dfs[0][1], True]

        if len(images_to_save_in_dfs)==3:
            train_df.loc[train_df.shape[0]] = [images_to_save_in_dfs[0][0].path,
                                                  images_to_save_in_dfs[0][0].id, images_to_save_in_dfs[0][1], False]
            train_df.loc[train_df.shape[0]] = [images_to_save_in_dfs[1][0].path,
                                                  images_to_save_in_dfs[1][0].id, images_to_save_in_dfs[1][1], False]
            validation_df.loc[validation_df.shape[0]]=[images_to_save_in_dfs[2][0].path,images_to_save_in_dfs[2][0].id
                ,images_to_save_in_dfs[2][1]]
            for i in range(24):
                train_df.loc[train_df.shape[0]] = [images_to_save_in_dfs[0][0].path,
                                                     images_to_save_in_dfs[0][0].id, images_to_save_in_dfs[0][1], True]
                train_df.loc[train_df.shape[0]] = [images_to_save_in_dfs[1][0].path,
                                                     images_to_save_in_dfs[1][0].id, images_to_save_in_dfs[1][1], True]



    pickle.dump(ImageInSet.name_to_id_dict,open("name_to_id_dict.pkl","wb"))

    pre_db_df=validation_df.append(train_df[train_df["to_aug"]==False].drop(["to_aug"],axis=1),ignore_index=True)
    db_df=pre_db_df.groupby(['class'],as_index=False).aggregate({'face_indexes':list,'path':list})
    db_df.to_csv(os.path.join(root_dir,'db.csv'),index=False)

    train_df.to_csv(os.path.join(root_dir,'train.csv'),index=False)
    validation_df.to_csv(os.path.join(root_dir,'validation.csv'),index=False)