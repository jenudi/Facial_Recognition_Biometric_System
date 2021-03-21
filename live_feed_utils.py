from images_classes import *
from aug import aug_img2
from torch.nn import functional as F
import torch
from DB_utils import *
from DB import db
import cv2 as cv
import os
import pickle
from datetime import datetime,date
from bson.son import SON


#name_to_id_dict = pickle.load(open("name_to_id_dict.pkl", "rb"))
id_to_name_dict_load = pickle.load(open("dict_cls2name.pickle", "rb"))


class LiveFeed:

    def __init__(self,db):
        self.db=db
        self.date=datetime.now().date()
        #self.number_of_employees=db.get_number_of_employees()
        self.id_to_name_dict = None
        self.number_of_employees=len(id_to_name_dict_load.keys())
        self.employees_entry_today = [False] * self.number_of_employees

    def update_employee_entry_today_by_db(self):
        entry_today_query_find = SON(
            {"date": SON({"year": self.date.year, "month": self.date.month, "day": self.date.day}),
             "entry": {"$ne": None}}), \
                                 SON({"employee id": 1, "_id": 0})
        entry_today_query = self.db.attendance_collection.find(entry_today_query_find[0], entry_today_query_find[1])
        for entry in entry_today_query:
            employee_id = entry["employee id"]
            self.employees_entry_today[employee_id - 1] = True

    def update_id_to_name_dict_by_db(self):
        employees_id_names_query=self.db.employees_collection.find({},{"_id":1,"name":1})
        self.id_to_name_dict=dict()
        for employee in employees_id_names_query:
            self.id_to_name_dict[employee["_id"]]=employee["name"]

    def register_entry(self, id_detected, date=None, time=datetime.now().time(), override=False):
        if date is None:
            date=self.date
        attendence_find = SON({"employee id": id_detected,"date": SON({"year": date.year,"month": date.month,"day": date.day})})
        attendence_query = self.db.attendance_collection.find(attendence_find)
        if override == False and len(list(attendence_query)) > 0:
            raise QueryError(' '.join(
                ["employee number", str(id_detected), "already registered entry at date", str(date) \
                 + "\nmust allow override in order to update entry"]))

        elif override and len(list(attendence_query)) > 0:
            date_query = {"employee id": id_detected,
                          "date": SON({"year": date.year, "month": date.month,"day": date.day})}
            update_entry = {"$set": {"entry": SON(
                {"hour": time.hour, "minute": time.minute,"second": time.second})}}
            self.db.attendance_collection.update_one(date_query, update_entry)
            print("".join(["database entry updated for employee id=" + str(id_detected)]))

        else:
            attendence_insert = SON({
                #"_id": '-'.join([str(id_detected), str(entry_date_and_time[0]), str(entry_date_and_time[1]),
                 #                str(entry_date_and_time[2])]),
                "_id":id_detected,
                "employee id": id_detected,
                "date": SON({"year": date.year, "month": date.month,
                             "day": date.day}),
                "entry": SON({"hour": time.hour, "minute": time.minute,
                              "second": time.second}),
                "exit": None,
                "total": None
            })
            self.db.attendance_collection.insert_one(attendence_insert)
            print("".join(["database entry registered for employee id=" + str(id_detected)]))


    def register_exit(self, id_detected, date=None, time=datetime.now().time(), override=False):
        if date is None:
            date=self.date
        entry_find = SON({"employee id": id_detected, "date": SON(
            {"year": int(date.year), "month": int(date.month), "day": int(date.day)}),
                          "entry": {"$ne": None}}), SON({"entry": 1, "_id": 0})
        entry_query = self.db.attendance_collection.find(entry_find[0], entry_find[1])
        entry_query_list = list(entry_query)
        if len(entry_query_list) == 0:
            raise QueryError(
                ' '.join(["employee id=", str(id_detected), "didn't register entry at date", str(date)]))

        attendance_doc_with_no_none_exit_find = SON({"employee id": id_detected,
                                                     "date": SON({"year": date.year,
                                                                  "month": date.month,
                                                                  "day": date.day}),
                                                     "exit": {"$ne": None}})
        attendance_doc_with_no_none_exit_query = self.db.attendance_collection.find(
            attendance_doc_with_no_none_exit_find)
        if override == False and len(list(attendance_doc_with_no_none_exit_query)) > 0:
            raise QueryError(' '.join(
                ["employee id=", str(id_detected), "already registered exit at date", str(date),
                 "\nmust allow override in order to update exit"]))

        else:
            date_query = {"employee id": id_detected,
                          "date": SON({"year": date.year, "month": date.month, "day": date.day})}
            update_exit = {"$set": {"exit": SON({"hour": time.hour, "minute": time.minute, "second": time.second})}}
            self.db.attendance_collection.update_one(date_query, update_exit)
            print("".join(["database exit updated for employee id=", str(id_detected)]))

            update_entry_date_and_time = [entry_query_list[0]["entry"]["hour"], entry_query_list[0]["entry"]["minute"],
                                          entry_query_list[0]["entry"]["second"]]
            update_exit_date_and_time = [time.hour, time.minute,time.second]
            hours, minutes, seconds = calculate_total(update_entry_date_and_time, update_exit_date_and_time)

            update_total = {"$set": {"total": SON({"hours": hours, "minutes": minutes, "seconds": seconds})}}
            self.db.attendance_collection.update_one(date_query, update_total)
            if len(list(attendance_doc_with_no_none_exit_query)) > 0:
                print("".join(["database total updated for employee id=", str(id_detected)]))
            else:
                print("".join(["database total registered for employee id=", str(id_detected)]))


os.chdir("C:\\Users\\gash5\\Desktop")


class CapturedFrame(ImageInSet):

    ANN_model = InceptionResnetV1(classify=True, pretrained='vggface2', num_classes=5748)
    ANN_model.load_state_dict(torch.load("model.pth",map_location=torch.device("cpu")))
    #KNN_model=pickle.load(open("knn_mode.pkl","rb"))
    face_recognition_threshold = 0
    number_of_faces_detected=0
    number_of_face_not_detected=0
    number_of_faces_recognized=0
    number_of_faces_not_recognized=0
    id_to_name_dict=id_to_name_dict_load
    #id_to_name_dict={value: key for key, value in name_to_id_dict.items()}

    def __init__(self,values):
        self.values=values
        self.name=""
        self.path=None
        self.face_image=None
        self.face_detected=False
        self.face_recognized=False
        self.id_detected=None
        self.recognition_probability=None

    def set_face_image(self):
        indexes_box=self.get_face_indexes()
        self.face_detected=True if (indexes_box is not None) and not (isinstance(indexes_box, type(None))) else False
        if self.face_detected:
            #self.save("face_image_temp.jpg")
            self.face_image = self.get_face_image(indexes_box)
            #os.remove("face_image_temp.jpg")
            #self.face_image=self.values[int(indexes_box[1]):int(indexes_box[3]), int(indexes_box[0]):int(indexes_box[2])]
            CapturedFrame.number_of_faces_detected+=1
        else:
            CapturedFrame.number_of_face_not_detected += 1

    def identify(self,model="knn"):
        if not self.face_detected:
            raise FrameException("face must be detected in order to perform identification")

        if model=="knn":
            '''
            self.face_image.augmentate()
            face_embedding=self.face_image.get_embedding(None)
            knn_id_prediction = CapturedFrame.KNN_model.predict(face_embedding)
            self.recognition_probability = CapturedFrame.KNN_model.predict_proba(face_embedding)[knn_id_prediction]
            print("recognition probability: " + str(self.recognition_probability))
            if self.recognition_probability>CapturedFrame.face_recognition_threshold:
                self.face_recognized = True
                self.id_detected = knn_id_prediction
                self.name=CapturedFrame.id_to_name_dict[self.id_detected]
            '''

        else:
            self.face_image.augmentate()
            norm_image = cv.normalize(self.face_image.values, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            tensor_img = torch.tensor(norm_image)
            resized_img = tensor_img.permute(2, 0, 1)
            unsqueezed_img=resized_img.unsqueeze(0)
            with torch.no_grad():
                CapturedFrame.ANN_model.eval()
                output = CapturedFrame.ANN_model(unsqueezed_img)
            self.recognition_probability=torch.max(F.softmax(output,dim=1),1)[0].item()
            print("recognition probability: " + str(self.recognition_probability))

            if self.recognition_probability>CapturedFrame.face_recognition_threshold:
                self.face_recognized = True
                self.id_detected = torch.max(F.softmax(output, dim=1), 1)[1].item()
                self.name=CapturedFrame.id_to_name_dict[self.id_detected]
                CapturedFrame.number_of_faces_recognized+=1
            else:
                CapturedFrame.number_of_faces_not_recognized+=1


class QueryError(Exception):
    pass


class FrameException(Exception):
    pass