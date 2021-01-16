from random import randint
from math import floor
from bson.son import SON
from pymongo import MongoClient
from sets_splits import db_df

def make_image_bson(path,embedding,Accuracy):
    return\
    SON({
        "file_name": path.split('\\')[-1],
        "path": path,
        "Accuracy": Accuracy,
        "embedding": SON({str(index):float(value) for index,value in enumerate(embedding)})
    })

def make_day_bson(year,month,day,entry_time=(8,0,0),exit_time=(17,0,0)):
    hours=exit_time[0] - entry_time[0] - (exit_time[1] < entry_time[1])
    minutes = exit_time[1] - entry_time[1] - (exit_time[2] < entry_time[2])
    if minutes<0:
        minutes=60+minutes
    seconds=exit_time[2] - entry_time[2]
    if seconds<0:
        seconds=60+seconds

    return \
    SON({
        "date": SON({"year":year,"month":month,"day":day}),
        "entry": SON({"hour":entry_time[0],"minute":entry_time[1],"second":entry_time[2]}),
        "exit": SON({"hour":exit_time[0],"minute":exit_time[1],"second":exit_time[2]}),
        "total": SON({"hours":hours,"minutes":minutes,"seconds":seconds })
    })

def get_random(list_of_values):
    rand_int=randint(1,100)
    if rand_int==100:
        return list_of_values[-1]
    devide_by_len=round(100/len(list_of_values))
    return list_of_values[floor(rand_int/devide_by_len)]


if __name__ == "__main__":

    employees=list()
    images=list()
    attendance=list()
    for index,name in enumerate(db_df['name']):

        employees.append(SON({
        "id": int(db_df.iloc[index]['id']),
        "employee number": int(db_df.iloc[index]['id']),
        "name": name,
        "branch": get_random(['A','B','C','D'])
        }))


        document_images=list()
        for embedding,path in zip(db_df.iloc[index]['embedding'],db_df.iloc[index]['path']):
            document_images.append(make_image_bson(path,embedding,None))

        images.append(SON({
            "id": int(db_df.iloc[index]['id']),
            "name": name,
            "images":document_images
        }))


        documents_attendance=[make_day_bson(2021,1,1),make_day_bson(2021,1,2,(8,randint(0,59),randint(0,59)),(17,randint(0,59),randint(0,59)))]

        attendance.append(SON({
            "id": int(db_df.iloc[index]['id']),
            "name": name,
            "attendence": documents_attendance
        }))



    client = MongoClient('mongodb://localhost:27017/')
    with client:
        biometric_system_db = client["biometric_system"]

        employees_collection = biometric_system_db["employees"]
        employees_collection.insert_many(employees)

        images_collection = biometric_system_db["images"]
        images_collection.insert_many(images)

        images_collection = biometric_system_db["attendance"]
        images_collection.insert_many(attendance)