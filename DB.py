from datetime import datetime
from random import randint
from math import floor
from bson.son import SON
from pymongo import MongoClient
from sets_splits import db_df


def make_image_doc(path,embedding,recognized="not yet tested",accuracy="not yet tested"):
    now=datetime.now().strftime('%Y %m %d %H %M %S').split(' ')
    return\
    SON({
        "_id": path,
        "recognized": recognized,
        "accuracy": accuracy,
        "embedding": list(map(float,embedding)),
        "uploaded": SON({"date": SON({"year":int(now[0]),"month":int(now[1]),"day":int(now[2])}),
                        "time":SON({"hour":int(now[3]),"minute":int(now[4]),"second":int(now[5])})})
    })


def make_attendance_doc(employee_id,year,month,day,entry_time=(8,0,0),exit_time=(17,0,0)):
    hours=exit_time[0] - entry_time[0] - (exit_time[1] < entry_time[1])
    minutes=exit_time[1] - entry_time[1] - (exit_time[2] < entry_time[2])
    if minutes<0:
        minutes+=60
    seconds=exit_time[2] - entry_time[2]
    if seconds<0:
        seconds+=60
    return\
    SON({
        "_id": '-'.join([str(employee_id),str(year),str(month),str(day)]),
        "employee id":employee_id,
        "date": SON({"year":year,"month":month,"day":day}),
        "entry": SON({"hour":entry_time[0],"minute":entry_time[1],"second":entry_time[2]}),
        "exit": SON({"hour":exit_time[0],"minute":exit_time[1],"second":exit_time[2]}),
        "total": SON({"hours":hours,"minutes":minutes,"seconds":seconds })
    })


def get_random(list_of_values):
    rand_int=randint(1,100)
    if rand_int==100:
        return list_of_values[-1]
    else:
        devide_by_len=round(100/len(list_of_values))
        return list_of_values[floor(rand_int/devide_by_len)]


if __name__ == "__main__":

    employees=list()
    images=list()
    attendance=list()

    for index,name in enumerate(db_df['name']):

        employee_id=int(db_df.iloc[index]['id'])

        employees.append(SON({
        "_id": employee_id,
        "employee number": employee_id,
        "name": name,
        "branch": get_random(['A','B','C','D']),
        "images paths": db_df.iloc[index]['path']
        }))

        for embedding,path in zip(db_df.iloc[index]['embedding'],db_df.iloc[index]['path']):
            images.append(make_image_doc(path,embedding))

        attendance.append(make_attendance_doc(employee_id,2021,1,1))
        attendance.append(make_attendance_doc(employee_id,2021,1,2,(8,randint(0,59),randint(0,59)),(17,randint(0,59),randint(0,59))))


    client = MongoClient('mongodb://localhost:27017/')
    with client:
        biometric_system_db = client["biometric_system"]

        employees_collection = biometric_system_db["employees"]
        employees_collection.insert_many(employees)

        images_collection = biometric_system_db["images"]
        images_collection.insert_many(images)

        images_collection = biometric_system_db["attendance"]
        images_collection.insert_many(attendance)