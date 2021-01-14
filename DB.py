from random import randint
from math import floor
from bson.son import SON
from pymongo import MongoClient
from sets_splits import db_df

def make_image_bson(path,embedding,Acuracy=None):
    return\
    SON({
        "file_name": path.split('\\')[-1],
        "path": path,
        "embedding": SON({str(index):float(value) for index,value in enumerate(embedding)}),
        "Acuracy": Acuracy
    })

def make_day_bson(year,month,day,entry_time=(8,0,0),exit_time=(17,0,0)):
    return\
    SON({
        "date": SON({"year":year,"month":month,"day":day}),
        "entry": SON({"hour":entry_time[0],"minute":entry_time[1],"second":entry_time[2]}),
        "exit": SON({"hour":exit_time[0],"minute":exit_time[1],"second":exit_time[2]})
    })

def get_random(list_of_values):
    rand_int=randint(1,100)
    if rand_int==100:
        return list_of_values[-1]
    devide_by_len=round(100/len(list_of_values))
    return list_of_values[floor(rand_int/devide_by_len)]


if __name__ == "__main__":

    employees=list()
    for index,person in enumerate(db_df['name']):

        document_images=list()
        for embedding,path in zip(db_df.iloc[index]['embedding'],db_df.iloc[index]['path']):
            document_images.append(make_image_bson(path,embedding))

        documents_attendence=[make_day_bson(2021,1,1),make_day_bson(2021,1,1,(8,randint(0,59),0),(17,randint(0,59),0))]

        employees.append(SON({
        "name": person,
        "id": index,
        "employee number": index,
        "branch": get_random(['A','B','C','D']),
        "attendence": documents_attendence,
        "images": document_images
        }))


    client = MongoClient('mongodb://localhost:27017/')
    with client:
        db = client["biometric_system"]
        employees_db=db["employees"]
        employees_db.insert_many(employees)