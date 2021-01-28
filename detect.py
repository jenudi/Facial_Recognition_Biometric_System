import cv2 as cv
import os
from images_classes import *
from images_sets_directories import train_paths
from facenet_embeddings import facenet_model


if __name__ == "__main__":

    captured_images_dir='\\'.join(os.getcwd(),'captured images')
    if not os.path.isdir(captured_images_dir):
        os.mkdir(captured_images_dir)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        frame = captured_frame(cv.resize(frame, None, fx=0.5, fy=0.5,interpolation=cv.INTER_AREA))
        if frame.face_detected:
            frame.face_image.resize_image()
            frame_face_embedding=frame.face_image.get_embedding(facenet_model,"normalize_by_train_values",facenet_model,train_paths)
            frame_title="Face detected"
        else:
            frame_title="No face detected"
        cv.imshow(frame_title, frame)
        c = cv.waitKey(1)
        if c == 27:
            break

    cap.release()
    #cv.waitKey()
    cv.destroyAllWindows()