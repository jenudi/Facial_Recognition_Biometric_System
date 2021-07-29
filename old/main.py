from image.ImageInSet import *
import pandas as pd
import os
import pickle


root_dir=os.getcwd()
directory_name = None
dataset_dir = os.path.join(root_dir, input("please enter the dataset directory path")) \
    if directory_name is None else os.path.join(root_dir, directory_name)
directories = [dir for dir in os.listdir(dataset_dir) if not '.' in dir]
train_df = pd.DataFrame(columns=['path','employee_id','face_indexes','number_to_augmante'])
validation_df = pd.DataFrame(columns=['path','employee_id','face_indexes'])
images_to_save_in_dfs = list()

for index,dir in enumerate(directories):
    images_to_save_in_dfs.clear()
    dir_path = os.path.join(dataset_dir,dir)
    files = os.listdir(dir_path)
    images_paths = [os.path.join(dir_path,file) for file in files if\
                    ((len(file.split('.')) == 2) and (file.split('.')[1] in ['jpg', 'jpeg','png']))]  # contains all the images of the person in the current directory

    for image_path in images_paths:
        if len(images_to_save_in_dfs)>=3:
            break
        print(image_path)
        image_object=ImageInSet(image_path)
        if image_object is None:
            continue
        image_face_indexes=image_object.get_face_indexes()
        if (not image_face_indexes is None) and (not isinstance(image_face_indexes, type(None))):
            images_to_save_in_dfs.append((image_object,image_face_indexes))

    if len(images_to_save_in_dfs)==0:
        del ImageInSet.name_to_id_dict[image_object.name]

    if len(images_to_save_in_dfs)==1:
        train_df.loc[train_df.shape[0]] = [images_to_save_in_dfs[0][0].path,images_to_save_in_dfs[0][0].employee_id, images_to_save_in_dfs[0][1], 20]

    if len(images_to_save_in_dfs)==2:
        train_df.loc[train_df.shape[0]] = [images_to_save_in_dfs[0][0].path,images_to_save_in_dfs[0][0].employee_id, images_to_save_in_dfs[0][1], 20]
        validation_df.loc[validation_df.shape[0]] = [images_to_save_in_dfs[1][0].path,images_to_save_in_dfs[1][0].employee_id,images_to_save_in_dfs[1][1]]

    if len(images_to_save_in_dfs)==3:
        train_df.loc[train_df.shape[0]] = [images_to_save_in_dfs[0][0].path,images_to_save_in_dfs[0][0].employee_id, images_to_save_in_dfs[0][1], 10]
        train_df.loc[train_df.shape[0]] = [images_to_save_in_dfs[1][0].path,images_to_save_in_dfs[1][0].employee_id, images_to_save_in_dfs[1][1], 10]

        validation_df.loc[validation_df.shape[0]]=[images_to_save_in_dfs[2][0].path,images_to_save_in_dfs[2][0].employee_id,images_to_save_in_dfs[2][1]]

pickle.dump(ImageInSet.name_to_id_dict,open("name_to_id_dict.pkl","wb"))

pre_db_df=validation_df.append(train_df,ignore_index=True)
db_df=pre_db_df.groupby(['employee_id'],as_index=False).aggregate({'face_indexes':list,'path':list})
db_df.to_pickle("db_df.pkl")

train_df.to_pickle("train_df.pkl")
validation_df.to_pickle("validation_df.pkl")

pickle.dump(ImageInSet.name_to_id_dict,open("name_to_id_dict.pkl","wb"))