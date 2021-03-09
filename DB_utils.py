from datetime import datetime
from random import randint
from math import floor
from bson.son import SON
from pymongo import MongoClient
from facenet_embeddings import db_df


def get_random(list_of_values):
    rand_int = randint(1, 100)
    if rand_int == 100:
        return list_of_values[-1]
    else:
        devide_by_len = round(100 / len(list_of_values))
        return list_of_values[floor(rand_int / devide_by_len)]


def calculate_total(entry_time, exit_time):
    hours = exit_time[0] - entry_time[0] - (exit_time[1] < entry_time[1])
    minutes = exit_time[1] - entry_time[1] - (exit_time[2] < entry_time[2])
    if minutes < 0:
        minutes += 60
    seconds = exit_time[2] - entry_time[2]
    if seconds < 0:
        seconds += 60
    return hours, minutes, seconds


def make_image_doc(path, employee_id, embedding, recognized="not yet tested", accuracy="not yet tested"):
    now = datetime.now().strftime('%Y %m %d %H %M %S').split(' ')
    return \
        SON({
            "_id": path.split('\\')[-1],
            "employee id": employee_id,
            "recognized": recognized,
            "accuracy": accuracy,
            "embedding": embedding,
            "uploaded": SON({"date": SON({"year": int(now[0]), "month": int(now[1]), "day": int(now[2])}),
                             "time": SON({"hour": int(now[3]), "minute": int(now[4]), "second": int(now[5])})})
        })



def make_attendance_doc(employee_id, year, month, day, entry_time=(8, 0, 0), exit_time=(17, 0, 0)):
    hours, minutes, seconds=calculate_total(entry_time, exit_time)
    return \
        SON({
            "_id": '-'.join([str(employee_id), str(year), str(month), str(day)]),
            "employee id": employee_id,
            "date": SON({"year": year, "month": month, "day": day}),
            "entry": SON({"hour": entry_time[0], "minute": entry_time[1], "second": entry_time[2]}),
            "exit": SON({"hour": exit_time[0], "minute": exit_time[1], "second": exit_time[2]}),
            "total": SON({"hours": hours, "minutes": minutes, "seconds": seconds})
        })



def make_employee_doc(employee_id, employee_number, name, images_directory_path,
                      branch=get_random(['A', 'B', 'C', 'D']), admin=False):
    SON({
        "_id": employee_id,
        "employee number": employee_id,
        "name": name,
        "images directory path": images_directory_path,
        "branch": branch,
        "admin": admin
    })


