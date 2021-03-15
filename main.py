import pandas as pd
import os


def make_df(directory_name=None):
    root_dir=os.getcwd()
    dataset_dir = os.path.join(root_dir, input("please enter the dataset directory name")) \
        if directory_name is None else os.path.join(root_dir, directory_name)
    directories = [dir for dir in os.listdir(dataset_dir) if not '.' in dir]
    df = pd.DataFrame(columns=['path','name','cls'])
    cls_num = 0
    for dir in directories:
        dir_path = os.path.join(dataset_dir,dir)
        images = [file for file in files if ((len(file.split('.')) == 2) and (file.split('.')[1] in ['jpg', 'jpeg','png']))]  # contains all the images of the person in the current directory
        for image_path in os.listdir(dir_path):
            if '.jpg' in file: # add other image file types
                df = df.append({'path': os.path.join(child,file),'name': dir,'cls': cls_num} ignore_index=True)
        cls_num += 1
    return df

