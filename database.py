from images_classes import *
import pandas as pd
import os
import pickle
from datetime import datetime
from random import randint
from math import floor
from bson.son import SON
from pymongo import MongoClient,errors


#name_to_id_dict = pickle.load(open("name_to_id_dict.pkl", "rb"))
#id_to_name_dict = {value: key for key, value in name_to_id_dict.items()}

#id_to_name_dict = pickle.load(open("dict_cls2name.pkl", "rb"))


class Database:

    MINIMUM_NUMBER_OF_IMAGES_FOR_MODEL = 4
    BRANCHES=["TA", "SF", "TO", "BN"]


    def __init__(self):
        self.__client = MongoClient('mongodb://localhost:27017/')
        with self.__client:
            self.__database = self.__client["biometric_system"]
            self.employees_collection = self.__database["employees"]
            self.images_collection = self.__database["images"]
            self.attendance_collection = self.__database["attendance"]


    def make_image_doc(self, path, employee_id, face_indexes,recognized_by_model=False):
        now = datetime.now()
        return \
            SON({
                "_id": path.split('\\')[-1],
                "employee id": employee_id,
                "face indexes": face_indexes,
                "recognized by model": recognized_by_model,
                "uploaded": SON({"date": SON({"year": now.year, "month": now.month, "day": now.day}),
                                 "time": SON({"hour": now.hour, "minute": now.minute, "second": now.second})})
            })


    def make_attendance_doc(self, employee_id, year, month, day, entry_time=(8,0,0), exit_time=(17,0,0)):
        hours, minutes, seconds = self.calculate_total(entry_time, exit_time)
        return \
            SON({
                "_id": '-'.join([str(employee_id), str(year), str(month), str(day)]),
                "employee id": employee_id,
                "date": SON({"year": year, "month": month, "day": day}),
                "entry": SON({"hour": entry_time[0], "minute": entry_time[1], "second": entry_time[2]}),
                "exit": SON({"hour": exit_time[0], "minute": exit_time[1], "second": exit_time[2]}),
                "total": SON({"hours": hours, "minutes": minutes, "seconds": seconds})
            })


    def make_employee_doc(self, employee_id, employee_number, name, images_directory_path,
                          number_of_images,branch=None,model_accuracy=None,
                          model_class=None,admin=False):
        if branch==None:
            branch=Database.BRANCHES[randint(0,3)]
        return \
            SON({
                "_id": employee_id,
                "employee number": employee_number,
                "name": name,
                "images directory path": images_directory_path,
                "branch": branch,
                "admin": admin,
                "number of images": number_of_images,
                "model accuracy": model_accuracy,
                "model class": model_class,
                "included in model":number_of_images>=Database.MINIMUM_NUMBER_OF_IMAGES_FOR_MODEL
            })


    def get_number_of_employees(self):
        return self.employees_collection.count_documents({})


    @staticmethod
    def calculate_total(entry_time, exit_time):
        hours = exit_time[0] - entry_time[0] - (exit_time[1] < entry_time[1])
        minutes = exit_time[1] - entry_time[1] - (exit_time[2] < entry_time[2])
        if minutes < 0:
            minutes += 60
        seconds = exit_time[2] - entry_time[2]
        if seconds < 0:
            seconds += 60
        return hours, minutes, seconds


database = Database()


if __name__ == "__main__":


    id_to_name_dict = pickle.load(open("dict_cls2name.pkl", "rb"))
    db_df = pickle.load(open(os.path.join(os.getcwd(), "db_df.pkl"), "rb"))

    employees = list()
    images = list()
    attendance = list()

    for index in range(db_df.shape[0]):

        # employee_id=int(db_df.iloc[index]['employee_id'])
        employee_id = index
        name = id_to_name_dict[index]

        employees.append(database.make_employee_doc(employee_id, employee_id, name,
                                                    '/'.join(db_df.iloc[index]['path'][0].split('\\')[:-1]),
                                                    int(db_df['pic_num'][index])))

        for path, face_indexes in zip(db_df.iloc[index]['path'], db_df.iloc[index]['indexes']):
            images.append(database.make_image_doc(path, employee_id, list(map(float, face_indexes))))

        attendance.append(database.make_attendance_doc(employee_id, 2021, 1, 1))
        attendance.append(database.make_attendance_doc(employee_id, 2021, 1, 2, (8, randint(0, 59), randint(0, 59)),
                                                       (17, randint(0, 59), randint(0, 59))))

    for employee_index in range(round(len(employees) / 10)):
        employees[employee_index]["admin"] = True

    database.employees_collection.insert_many(employees)
    database.images_collection.insert_many(images)
    database.attendance_collection.insert_many(attendance)