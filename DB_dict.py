from DB_utils import *
from DB import db


if __name__ == "__main__":

    employees=list()
    attendance=list()


    os.chdir("C:\\Users\\gash5\\Desktop")
    dict_cls2name = pickle.load(open("dict_cls2name.pickle", "rb"))


    for key in dict_cls2name.keys():
        employee_id = key
        print(employee_id)
        name = dict_cls2name[key]

        employees.append(db.make_employee_doc(employee_id, employee_id, name, ""))
        attendance.append(db.make_attendance_doc(employee_id, 2021, 1, 1))
        attendance.append(db.make_attendance_doc(employee_id, 2021, 1, 2, (8, randint(0, 59), randint(0, 59)),
                                                 (17, randint(0, 59), randint(0, 59))))


        for employee_index in range(round(len(employees) / 10)):
            employees[employee_index]["admin"] = True


        db.employees_collection.insert_many(employees)
        db.attendance_collection.insert_many(attendance)