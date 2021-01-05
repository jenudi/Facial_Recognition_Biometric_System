from numpy import expand_dims
from numpy import load #enables loading npz file
from numpy import asarray
import tensorflow as tf


#get face embedding for one face

def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32') #scale pixel values
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std #standardize pixel values
    samples = expand_dims(face_pixels, axis=0) #transform face into one sample
    yhat = model.predict(samples) #make prediction to get embedding
    return yhat[0]

#loading face dataset
faces_dataset_path = "face_dataset_numpy.npz"
data = load(faces_dataset_path)
faces, labels = data['arr_0'], data['arr_1']
print('Loaded: ', faces.shape, labels.shape)

#load facenet model
model = tf.keras.models.load_model('facenet_keras.h5')
print('Loaded Model')

#convert each face to embedding
face_embeddings = list()
for face_pixels in faces:
    embedding = get_embedding(model, face_pixels)
    face_embeddings.append(embedding)
face_embeddings = asarray(face_embeddings)
print(face_embeddings.shape)

#save arrays to one compressed file
np.savez_compressed('face_embeddings.npz', face_embeddings, labels)