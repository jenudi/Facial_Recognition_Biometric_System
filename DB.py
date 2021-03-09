from DB_utils import *

if __name__ == "__main__":

    employees=list()
    images=list()
    attendance=list()

    for index,name in enumerate(db_df['name']):

        employee_id=int(db_df.iloc[index]['id'])

        employees.append(make_employee_doc(employee_id,employee_id,name,'/'.join(db_df.iloc[index]['path'][0].split('\\')[:-1])))

        for embedding,path in zip(db_df.iloc[index]['embedding'],db_df.iloc[index]['path']):
            images.append(make_image_doc(path,employee_id,embedding))

        attendance.append(make_attendance_doc(employee_id,2021,1,1))
        attendance.append(make_attendance_doc(employee_id,2021,1,2,(8,randint(0,59),randint(0,59)),(17,randint(0,59),randint(0,59))))


    for employee_index in range(round(len(employees)/10)):
        employees[employee_index]["admin"]=True


    client = MongoClient('mongodb://localhost:27017/')
    with client:
        biometric_system_db = client["biometric_system"]

        employees_collection = biometric_system_db["employees"]
        employees_collection.insert_many(employees)

        images_collection = biometric_system_db["images"]
        images_collection.insert_many(images)

        attendance_collection = biometric_system_db["attendance"]
        attendance_collection.insert_many(attendance)