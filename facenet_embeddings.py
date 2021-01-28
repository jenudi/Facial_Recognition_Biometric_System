import pandas as pd
from tensorflow.keras.models import load_model
from images_sets_directories import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_embedding(cur_image, normalization_method, model, train_paths_list=None):
    if normalization_method=="normalize_by_train_values":
        assert (not train_paths_list is None) or (not isinstance(face, type(None))), "enter train paths list in order to use the normalize by train values method"
        norm_values = cur_image.normalize_by_train_values(train_paths_list).astype("float32")
    else:
        norm_values = cur_image.normalize_by_image_values()
    four_dim_values = np.expand_dims(norm_values, axis=0)
    embedding=model.predict(four_dim_values)[0]
    return embedding


os.chdir(dataset_dir)

facenet_model = load_model('facenet_keras.h5',compile=False)

#all the images in the train, validation and test sets go through normalization
#the normalization type is standardization that is done by substracting the train set mean and dividing by the train set STD
#the normalized values are calculated by the normalize mothod
train_df=pd.DataFrame(columns=['id', 'name', 'embedding', 'path'])
for index,image_paths in enumerate(train_paths):
    cur_image = image_in_set(image_paths[0])
    cur_image_embedding=get_embedding(cur_image,"normalize_by_train_values",facenet_model,train_paths)
    train_df.loc[index]=[cur_image.id, cur_image.name, cur_image_embedding, image_paths[1]]

augmentation_df=pd.DataFrame(columns=['id', 'name', 'embedding'])
for index,image_path in enumerate(augmentation_paths):
    cur_image = image_in_set(image_path)
    cur_image_embedding=get_embedding(cur_image,"normalize_by_train_values",facenet_model,train_paths)
    augmentation_df.loc[index]=[cur_image.id, cur_image.name, cur_image_embedding]

validation_df=pd.DataFrame(columns=['id', 'name', 'embedding', 'path'])
for index,image_paths in enumerate(validation_paths):
    cur_image = image_in_set(image_paths[0])
    cur_image_embedding=get_embedding(cur_image,"normalize_by_train_values",facenet_model,train_paths)
    validation_df.loc[index]=[cur_image.id, cur_image.name, cur_image_embedding, image_paths[1]]

test_df=pd.DataFrame(columns=['id', 'name', 'embedding', 'path'])
for index,image_paths in enumerate(test_paths):
    cur_image = image_in_set(image_paths[0])
    cur_image_embedding=get_embedding(cur_image,"normalize_by_train_values",facenet_model,train_paths)
    test_df.loc[index]=[cur_image.id, cur_image.name, cur_image_embedding, image_paths[1]]

all_data_df = pd.concat([train_df,validation_df,test_df],ignore_index=True)
db_df=all_data_df.groupby(['id','name'],as_index=False).aggregate({'embedding':list, 'path':list})

train_df=pd.concat([train_df.drop('path',axis=1),augmentation_df],ignore_index=True)
validation_df.drop('path',axis=1, inplace=True)
test_df.drop('path',axis=1, inplace=True)

cvs_dir=''.join([dataset_dir,'\\sets_csv_files'])
if not os.path.isdir(cvs_dir):
    os.mkdir(cvs_dir)
os.chdir(cvs_dir)
train_df.to_csv(''.join([os.getcwd(),'\\train.csv']),index=False)
validation_df.to_csv(''.join([os.getcwd(),'\\validation.csv']),index=False)
test_df.to_csv(''.join([os.getcwd(),'\\test.csv']),index=False)

os.chdir('../../')