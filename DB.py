import numpy as np
import pandas as pd
import random
from datetime import datetime
import pymongo
import bson
from datetime import date,time
from pymongo import MongoClient
from sets_splits import db_df


if __name__ == "__main__":

    documents_for_db=list()
    for index,person in enumerate(db_df['name']):
        document_images=list()
        for embedding,path in zip(db_df.iloc[index]['embedding'],db_df.iloc[index]['path']):
            document_images.append(bson.son.SON({
                "file_name" : path.split('\\')[-1],
                "path" : path,
                "embedding":embedding,
                "Acuracy":None
            }))
        documents_attendence=[
            bson.son.SON({
                "Date":date(2021,1,1),
                "Entry":time(8,0,0),
                "Exit":time(17,0,0)
            }),
            bson.son.SON({
                "Date": date(2021, 1, 2),
                "Entry": time(8, 30, 0),
                "Exit": time(16, 30, 0)
            })
        ]
        documents_for_db.append(bson.son.SON({
        "Name":person,
        "ID":index,
        "Employee number":index,
        "Branch":random.choise(['A','B','C','D']),
        "Attendence":documents_attendence,
        "Images":document_images
        }))
    
    client = MongoClient('mongodb://localhost:27017/')
    with client:
        db = client.biometric_system
        db.faces.insert_many(documents_for_db)