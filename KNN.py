from images_classes import *
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os


class KnnModelArgs:

    def __init__(self):
        self.n_neighbors=range(5,21,5)
        self.algorithms=["auto", "ball_tree", "kd_tree", "brute"]
        self.weights=["uniform", "distance"]
        self.metrics=["euclidean","manhattan"]
        self.norm_embeddings=[True,False]



root_dir=os.getcwd()

train_df=pickle.load(open(os.path.join(root_dir,'train.pkl'),"rb"))
validation_df=pickle.load(open(os.path.join(root_dir,'validation.pkl'),"rb"))

train_embeddings=list()
train_ids=list()

for index,image in train_df.iterrows():
    print("train "+ str(index))
    train_ids.append(image["employee_id"])
    new_face_image=ImageInSet(image["path"]).get_face_image(image["face_indexes"])
    new_face_image.augmentate()
    train_embeddings.append(new_face_image.get_embedding(None))
    for i in range(image["number_to_augmante"]):
        train_ids.append(image["employee_id"])
        new_face_image = ImageInSet(image["path"]).get_face_image(image["face_indexes"])
        new_face_image.augmentate("train")
        train_embeddings.append(new_face_image.get_embedding(None))


validation_embeddings=list()
validation_ids=list()
for index,image in validation_df.iterrows():
    print("validation "+ str(index))
    validation_ids.append(image["employee_id"])
    new_face_image=ImageInSet(image["path"]).get_face_image(image["face_indexes"])
    new_face_image.augmentate("validation")
    validation_embeddings.append(new_face_image.get_embedding(None))


pickle.dump(train_embeddings, open("train_embeddings.pkl", "wb"))
pickle.dump(train_embeddings, open("train_ids.pkl", "wb"))
pickle.dump(train_embeddings, open("validation_embeddings.pkl", "wb"))
pickle.dump(train_embeddings, open("validation_ids.pkl", "wb"))


knn_model_args=KnnModelArgs()
for n_neighbors_number in knn_model_args.n_neighbors:
    for algorithm in knn_model_args.algorithms:
        for weights_type in knn_model_args.weights:
            for metric in knn_model_args.metrics:
                for norm in knn_model_args.norm_embeddings:

                    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors_number,algorithm=algorithm,weights=weights_type,metric=metric)

                    if norm:
                        train_embeddings_for_knn=[embedding/np.linalg.norm(embedding) for embedding in train_embeddings]
                        validation_embeddings_for_knn=[embedding/np.linalg.norm(embedding) for embedding in validation_embeddings]
                    else:
                        train_embeddings_for_knn=train_embeddings
                        validation_embeddings_for_knn=validation_embeddings

                    knn_model.fit(train_embeddings_for_knn, train_ids)
                    knn_score = knn_model.score(validation_embeddings_for_knn, validation_ids)

                    model_description="n_neighbors="+ str(n_neighbors_number)+ " algorithm="+algorithm+" weights="+ weights_type+ " metric:"+metric+ " norm="+str(norm)+" score:" + str(knn_score)
                    print(model_description)

                    pickle.dump(knn_model,open(model_description+".pkl","wb"))