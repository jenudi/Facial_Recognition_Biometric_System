from random import randint


class Employee:

    BRANCHES=["TA", "SF", "TO", "BN"]

    def __init__(self,id,employee_number,name,images_directory_path, number_of_images):
        self.id=id
        self.employee_number=employee_number
        self.name=name
        self.images_directory_path=images_directory_path
        self.number_of_images=number_of_images
        self.branch=Employee.get_random_branch()
        self.model_accuracy=0.0
        self.model_class=-1
        self.admin=False

    @classmethod
    def get_random_branch(cls):
        return cls.BRANCHES[randint(0,3)]