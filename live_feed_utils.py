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


class QueryError(Exception):
    pass


def register_entry(id_detected,entry_date_and_time,attendance_collection,override=False):
    attendence_find=SON({"employee id": id_detected,
                                                       "date": SON({"year": int(entry_date_and_time[0]),
                                                                    "month": int(entry_date_and_time[1]),
                                                                    "day": int(entry_date_and_time[2])})})
    attendence_query = attendance_collection.find(attendence_find)
    if override==False and len(list(attendence_query))>0:
        raise QueryError(' '.join(["employee number", str(id_detected), "already registered entry at date", str(entry_date_and_time[0]), str(entry_date_and_time[1]), str(entry_date_and_time[2])\
                       + "\nmust allow override in order to update entry"]))

    elif override and len(list(attendence_query))>0:
        date_query={"employee id": id_detected,
                    "date": SON({"year": int(entry_date_and_time[0]), "month": int(entry_date_and_time[1]), "day": int(entry_date_and_time[2])})}
        update_entry={"$set":{"entry":SON({"hour": int(entry_date_and_time[3]), "minute": int(entry_date_and_time[4]), "second": int(entry_date_and_time[5])})}}
        attendance_collection.update_one(date_query,update_entry)
        print("".join(["database entry updated for employee id="+ str(id_detected)]))

    else:
        attendence_insert=SON({
            "_id": '-'.join([str(id_detected),str(entry_date_and_time[0]),str(entry_date_and_time[1]),str(entry_date_and_time[2])]),
            "employee id": id_detected,
            "date": SON({"year": int(entry_date_and_time[0]), "month": int(entry_date_and_time[1]), "day": int(entry_date_and_time[2])}),
            "entry": SON({"hour": int(entry_date_and_time[3]), "minute": int(entry_date_and_time[4]), "second": int(entry_date_and_time[5])}),
            "exit": None,
            "total": None
        })
        attendance_collection.insert_one(attendence_insert)
        print("".join(["database entry registered for employee id="+ str(id_detected)]))


def register_exit(id_detected,exit_date_and_time,attendance_collection,override=False):
    entry_find=SON({"employee id": id_detected, "date": SON({"year": int(exit_date_and_time[0]), "month": int(exit_date_and_time[1]), "day": int(exit_date_and_time[2])}),
                    "entry":{"$ne": None}}),SON({"entry":1,"_id":0})
    entry_query=attendance_collection.find(entry_find[0],entry_find[1])
    entry_query_list=list(entry_query)
    if len(entry_query_list)==0:
        raise QueryError(' '.join(["employee id=", str(id_detected), "didn't register entry at date", str(exit_date_and_time[0]), str(exit_date_and_time[1]), str(exit_date_and_time[2])]))

    attendance_doc_with_no_none_exit_find=SON({"employee id": id_detected,
                                                 "date": SON({"year": int(exit_date_and_time[0]),
                                                              "month": int(exit_date_and_time[1]),
                                                              "day": int(exit_date_and_time[2])}),
                                                 "exit": {"$ne": None}})
    attendance_doc_with_no_none_exit_query = attendance_collection.find(attendance_doc_with_no_none_exit_find)
    if override==False and len(list(attendance_doc_with_no_none_exit_query))>0:
        raise QueryError(' '.join(["employee id=", str(id_detected), "already registered exit at date", str(exit_date_and_time[0]), str(exit_date_and_time[1]), str(exit_date_and_time[2]), \
                        "must allow override in order to update exit"]))

    else:
        date_query={"employee id": id_detected,
                    "date": SON({"year": int(exit_date_and_time[0]), "month": int(exit_date_and_time[1]), "day": int(exit_date_and_time[2])})}
        update_exit={"$set":{"exit":SON({"hour": int(exit_date_and_time[3]), "minute": int(exit_date_and_time[4]), "second": int(exit_date_and_time[5])})}}
        attendance_collection.update_one(date_query,update_exit)
        print("".join(["database exit updated for employee id=",str(id_detected)]))

        update_entry_date_and_time=[entry_query_list[0]["entry"]["hour"],entry_query_list[0]["entry"]["minute"],entry_query_list[0]["entry"]["second"]]
        update_exit_date_and_time=[int(exit_date_and_time[3]),int(exit_date_and_time[4]),int(exit_date_and_time[5])]
        hours, minutes, seconds = calculate_total(update_entry_date_and_time,update_exit_date_and_time)

        update_total = {"$set": {"total": SON({"hours": hours, "minutes": minutes, "seconds": seconds})}}
        attendance_collection.update_one(date_query, update_total)
        if len(list(attendance_doc_with_no_none_exit_query)) > 0:
            print("".join(["database total updated for employee id=", str(id_detected)]))
        else:
            print("".join(["database total registered for employee id=",str(id_detected)]))