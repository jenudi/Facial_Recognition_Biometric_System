from images_classes import *
import pickle
import os


train_df=pickle.load(open("train.pkl","rb"))
validation_df=pickle.load(open("validation.pkl","rb"))

train_embeddings=list()
train_ids=list()

for index,image in train_df.iterrows():
    print("train "+ str(index))
    train_ids.append(image["employee_id"])
    new_face_image=ImageInSet(image["path"]).get_face_image(image["face_indexes"])
    new_face_image.augmentate()
    train_embeddings.append(np.array(new_face_image.get_embedding(None)))
    for i in range(image["number_to_augmante"]):
        train_ids.append(image["employee_id"])
        new_face_image = ImageInSet(image["path"]).get_face_image(image["face_indexes"])
        new_face_image.augmentate("train")
        train_embeddings.append(np.array(new_face_image.get_embedding(None)))


validation_embeddings=list()
validation_ids=list()
for index,image in validation_df.iterrows():
    print("validation "+ str(index))
    validation_ids.append(image["employee_id"])
    new_face_image=ImageInSet(image["path"]).get_face_image(image["face_indexes"])
    new_face_image.augmentate("validation")
    validation_embeddings.append(np.array(new_face_image.get_embedding(None)))


pickle.dump(train_embeddings, open("train_embeddings.pkl", "wb"))
pickle.dump(train_ids, open("train_ids.pkl", "wb"))
pickle.dump(validation_embeddings, open("validation_embeddings.pkl", "wb"))
pickle.dump(validation_ids, open("validation_ids.pkl", "wb"))