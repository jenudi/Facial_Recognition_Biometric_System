from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pickle


train_embeddings=pickle.load(open("train_embeddings.pkl", "rb"))
train_ids=pickle.load(open("train_ids.pkl", "rb"))
validation_embeddings=pickle.load(open("validation_embeddings.pkl", "rb"))
validation_ids=pickle.load(open("validation_ids.pkl", "rb"))


class KnnModelArgs:

    def __init__(self):
        self.n_neighbors=[1,3,5,6,7]
        self.algorithms=["auto", "brute"]
        self.weights=["uniform", "distance"]
        self.metrics=["euclidean","manhattan"]


knn_model_args=KnnModelArgs()
models=list()
scores=list()
for n_neighbors_number in knn_model_args.n_neighbors:
    for algorithm in knn_model_args.algorithms:
        for weights_type in knn_model_args.weights:
            for metric in knn_model_args.metrics:

                models.append([n_neighbors_number, algorithm, weights_type, metric])

                knn_model = KNeighborsClassifier(n_neighbors=n_neighbors_number,algorithm=algorithm,weights=weights_type,metric=metric)

                knn_model.fit(train_embeddings, train_ids)
                knn_score = knn_model.score(validation_embeddings, validation_ids)

                scores.append(knn_score)

                print("model number "+ str(len(models)-1)+ " n_neighbors:"+ str(n_neighbors_number)+ " algorithm:"+algorithm+" weights:"+ weights_type+ " metric:"+metric +" score:" + str(knn_score))


best_score=max(scores)
best_model=models[np.argmax(scores)]

print("the best model is "+ str(best_model) + " with a score of " +str(best_score))

knn_model = KNeighborsClassifier(n_neighbors=best_model[0], algorithm=best_model[1], weights=best_model[2],metric=best_model[3])

embeddings_for_final_model=sum([train_embeddings,validation_embeddings],[])
ids_for_final_model=sum([train_ids,validation_ids],[])

knn_model.fit(embeddings_for_final_model, ids_for_final_model)

pickle.dump(knn_model,open("knn_model.pkl","wb"))