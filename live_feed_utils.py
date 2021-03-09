import cv2 as cv
import os
import pickle
from pymongo import MongoClient
from images_classes import *
from DB_utils import *
from datetime import datetime
from bson.son import SON


client = MongoClient('mongodb://localhost:27017/')
with client:
    biometric_system_db = client["biometric_system"]
    attendance_collection = biometric_system_db["attendance"]


id_to_name_dict={value:key for key,value in Image_in_set.name_to_id_dict.items()}

train_paths=pickle.load(open("train_paths.pkl","rb"))


def register_entry(id_detected,date_and_time,attendance_collection):
    attendence_query=attendance_collection.find(SON({"employee id": id_detected,
    "date": SON({"year": int(date_and_time[0]), "month": int(date_and_time[1]), "day": int(date_and_time[2])})}))

    if len(list(attendence_query))>0:
        print(' '.join(["employee number", str(id_detected), "already registered entry at date", date_and_time[0], date_and_time[1], date_and_time[2]]))
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
        print("database updated for id "+ str(id_detected))


def register_exit(id_detected,exit_date_and_time,attendance_collection,override=False):
    entry_query=attendance_collection.find(SON({"employee id": id_detected,
    "date": SON({"year": int(exit_date_and_time[0]), "month": int(exit_date_and_time[1]), "day": int(exit_date_and_time[2])}),
    "entry":{"$ne": None}}),{"entry":1,"_id":0})
    if len(list(entry_query))==0:
        print(' '.join(["employee number", id_detected, "didn't register entry at date", exit_date_and_time[0], exit_date_and_time[1], exit_date_and_time[2]]))
        return

    if override:
        exit_query=attendance_collection.find(SON({"employee id": id_detected,
        "date": SON({"year": int(exit_date_and_time[0]), "month": int(exit_date_and_time[1]), "day": int(exit_date_and_time[2])}),
        "exit":{"$ne": None}}))
            if len(list(exit_query))>0:
                print(' '.join(["employee number", id_detected, "already registered exit at date", exit_date_and_time[0], exit_date_and_time[1], exit_date_and_time[2]]))
                return

    date_query={"employee id": id_detected,
                "date": SON({"year": int(exit_date_and_time[0]), "month": int(exit_date_and_time[1]), "day": int(exit_date_and_time[2])})}
    update_exit={"$set":{"exit":SON({"hour": int(exit_date_and_time[3]), "minute": int(exit_date_and_time[4]), "second": int(exit_date_and_time[5])})}}
    attendance_collection.update_one(date_query,update_exit)
    hours, minutes, seconds = calculate_total([entry_query["entry"]["hour"],entry_query["entry"]["minute"],entry_query["entry"]["second"]],\
                                                [exit_date_and_time[3],exit_date_and_time[4],exit_date_and_time[5]])
    update_total = {"$set": {"total": SON({"hour": hours, "minute": minutes, "second": seconds})}}
    attendance_collection.update_one(date_query, update_total)