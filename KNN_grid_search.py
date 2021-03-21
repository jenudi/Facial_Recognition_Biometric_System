from sklearn.neighbors import KNeighborsClassifier
import pickle


train_embeddings=pickle.load(open("train_embeddings.pkl", "rb"))
train_ids=pickle.load(open("train_ids.pkl", "rb"))
validation_embeddings=pickle.load(open("validation_embeddings.pkl", "rb"))
validation_ids=pickle.load(open("validation_ids.pkl", "rb"))


class KnnModelArgs:

    def __init__(self):
        self.n_neighbors=[5,10]
        self.algorithms=["auto", "ball_tree"]
        self.weights=["uniform", "distance"]
        self.metrics=["euclidean","manhattan"]
        self.norm_embeddings=[True,False]


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

