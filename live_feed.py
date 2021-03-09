import cv2 as cv
import os
import pickle
from tensorflow.keras.models import load_model
from pymongo import MongoClient
from images_classes import *
from datetime import datetime
from bson.son import SON


client = MongoClient('mongodb://localhost:27017/')
with client:
    biometric_system_db = client["biometric_system"]
    attendance_collection = biometric_system_db["attendance"]


id_to_name_dict={value:key for key,value in Image_in_set.name_to_id_dict.items()}

train_paths=pickle.load(open("train_paths.pkl","rb"))


identification_threshold=9.0


def register_entry(id_detected,date_and_time,attendance_collection):
    attendence_query=attendance_collection.find(SON({"employee id": id_detected,
    "date": SON({"year": int(date_and_time[0]), "month": int(date_and_time[1]), "day": int(date_and_time[2])})}))
    if len(attendence_query)>0:
        print(' '.join(["employee number", id_detected, "already registered entry at date", date_and_time[0], date_and_time[1], date_and_time[2]]))
    else:
        attendence_insert=SON({
            "_id": '-'.join([str(id_detected),str(date_and_time[0]),str(date_and_time[1]),str(date_and_time[2])]),
            "employee id": id_detected,
            "date": SON({"year": int(date_and_time[0]), "month": int(date_and_time[1]), "day": int(date_and_time[2])}),
            "entry": SON({"hour": int(date_and_time[3]), "minute": int(date_and_time[4]), "second": int(date_and_time[5])}),
            "exit": None,
            "total": None
        })
        attendance_collection.insert_one(attendence_insert)


def register_exit(id_detected,date_and_time,attendance_collection):

    entry_query=attendance_collection.find(SON({"employee id": id_detected,
    "date": SON({"year": int(date_and_time[0]), "month": int(date_and_time[1]), "day": int(date_and_time[2])}),
    "entry":{"$ne": None}}))
    exit_query=attendance_collection.find(SON({"employee id": id_detected,
    "date": SON({"year": int(date_and_time[0]), "month": int(date_and_time[1]), "day": int(date_and_time[2])}),
    "exit":{"$ne": None}}))

    if len(entry_query)==0:
        print(' '.join(["employee number", id_detected, "didn't register entry at date", date_and_time[0], date_and_time[1], date_and_time[2]]))
    elif len(exit_query)>0:
        print(' '.join(["employee number", id_detected, "already registered exit at date", date_and_time[0], date_and_time[1], date_and_time[2]]))
    else:
        date_query={"employee id": id_detected,
                    "date": SON({"year": int(date_and_time[0]), "month": int(date_and_time[1]), "day": int(date_and_time[2])})}
        update_exit={"$set":{"exit":SON({"hour": int(date_and_time[3]), "minute": int(date_and_time[4]), "second": int(date_and_time[5])})}}
        attendance_collection.update_one(date_query,update_exit)


if __name__ == "__main__":

    captured_images_dir='\\'.join([os.getcwd(),'captured images'])
    if not os.path.isdir(captured_images_dir):
        os.mkdir(captured_images_dir)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    face_detected_number=0
    while True:
        ret, frame = cap.read()
        frame = Captured_frame(cv.resize(frame, None, fx=0.5, fy=0.5,interpolation=cv.INTER_AREA))
        frame.set_face_image()
        if frame.face_detected:
            frame.face_image.resize_image()
            face_detected_number+=1
            frame.save("".join([captured_images_dir,"\\",str(face_detected_number),".jpg"]))
            frame.identify("normalize_by_train_values",train_paths)
            if frame.identification_probability>identification_threshold:
                now = datetime.now().strftime('%Y %m %d %H %M %S').split(' ')
                register_entry(frame.id_detected,now,attendance_collection)
            frame_title="Face detected"
        else:
            frame_title="No face detected"
        cv.imshow(frame_title, frame.values)
        c = cv.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv.destroyAllWindows()