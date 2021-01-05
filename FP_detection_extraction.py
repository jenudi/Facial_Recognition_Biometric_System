#!pip install mtcnn
from os import listdir
from os.path import isdir
import os
import numpy as np
import cv2 as cv
from mtcnn.mtcnn import MTCNN

from PIL import Image
from matplotlib import pyplot as plt
import time
from pathlib import Path

def extract_face(path_to_filename, detector, required_size=(160,160), save_faces=True):

  img = cv.imread(path_to_filename)
  img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
  pixels = np.asarray(img)
  results = detector.detect_faces(pixels) #detect faces in the image
  x1, y1, width, height = results[0]['box'] #extract the bounding box from the first face
  x1, y1 = abs(x1), abs(y1) #bug fix, to deal with negative numbers
  x2, y2 = x1+width, y1+height
  face = pixels[y1:y2, x1:x2]
  img = cv.resize(required_size)
  #image = image.fromarray(face) #convert from array back to image
  #image =  #resize to the model size

  if (save_faces):
    path = os.path.split(os.path.abspath(path_to_filename))[0]
    file_name = os.path.split(os.path.abspath(path_to_filename))[1]
    person_name = os.path.basename(os.path.normpath(Path(path)))
    project_folder = Path(path).parent.parent #goes to grandparent folder before creating new folder
    print(person_name)
    target_folder = os.path.join(project_folder, 'faces_mini_datadet', person_name)
    if not os.path.exists(target_folder):
      os.makedirs(target_folder)
    target_face_file_path = os.path.join(target_folder, file_name)
    print(target_face_file_path)
    image.save(target_face_file_path)
  face_array = asarray(image)
  return face_array

def extract_faces(directory):
  print('load faces')
  faces = list()

  detector = MTCNN()
  print('Extracting faces from ', directory, '...')
  for filename in listdir(directory):
    path = directory + filename
    try:
      face = extract_face(path, detector, save_faces=True)
    except Exception as e:
      continue
    faces.append(face)
  return faces

def generate_faces_from_images(directory):
  print('Load dataset...')
  X, y = list(), list()
  num = 1
  for subdir in listdir(directory):
    path = directory + subdir + '/'
    if not isdir(path):
      continue
    faces = extract_faces(path) #load all faces in subdirectory
    labels = [subdir for _ in range(len(faces))] #create labels

    print('> %d) loaded %d examples for class: %s' % (num, len(faces), subdir))
    num += 1
    X.extend(faces)
    y.extend(labels)
  return asarray(X), asarray(y)

faces, labels = generate_faces_from_images('/content/data_images/') #creates faces and labels numpy arrays
print(faces.shape, labels.shape)
np.savez_compressed("face_dataset_numpy.npz", faces, labels) #saves arrays to file for accessing later
