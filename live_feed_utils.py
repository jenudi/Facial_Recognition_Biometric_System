from images_classes import *
from DB_utils import *
import cv2 as cv
import os
import pickle
from datetime import datetime,date
from bson.son import SON


class Live_feed:

    def __init__(self,db,time_zone):
        self.db=db
        self.date=date.today().date()
        employees_entry_today = [False] * db.get_number_of_employees()
        id_to_name_dict = {value: key for key, value in Image_in_set.name_to_id_dict.items()}

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
        attendence_find = SON({"employee id": id_detected,"date": SON({"year": date.year,"month": date.month,"day": date.minute})})
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



class Captured_frame(Image_in_set):

    number_of_faces_detected=0
    number_of_face_not_detected=0
    number_of_faces_recognized=0
    number_of_faces_not_recognized=0

    def __init__(self,values):
        self.values=values
        self.name=None
        self.path=None
        self.face_image=None
        self.face_detected=False
        self.face_recognized=False
        self.id_detected=None
        self.recognition_probability=None

    def set_face_image(self):
        self.face_image=self.get_face_image()
        self.face_detected=True if (self.face_image is not None) and not (isinstance(self.face_image, type(None))) else False
        if self.face_detected:
            Captured_frame.number_of_faces_detected+=1
        else:
            Captured_frame.number_of_face_not_detected += 1

    def identify(self,normalize_method,train_paths,id_to_name_dict):
        if not self.face_detected:
            raise FrameException("face must be detected in order to perform identification")
        face_embedding = self.face_image.get_embedding(normalize_method, train_paths)
        self.identification_probability = uniform(0.8,1.0)
        if self.identification_probability>Image_in_set.face_recognition_threshold:
            self.face_recognized = True
            Captured_frame.number_of_faces_recognized+=1
            self.id_detected = randint(2, Captured_frame.number_of_employees)
        else:
            Captured_frame.number_of_faces_not_recognized+=1


train_paths=pickle.load(open("train_paths.pkl","rb"))

class QueryError(Exception):
    pass