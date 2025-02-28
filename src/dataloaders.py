import os
from glob import glob
from typing import List, Tuple


class DataReader:
    classes: List = []
    paths: List = []
    def __init__(self,dir:str):
        if not os.path.isdir(dir):
            error_message = f"{dir} is not exist"
            raise ValueError(error_message)
        self.dir = dir


    def to_category(self)->Tuple[List[str],List[str]]:

        for dirname, _, filenames in os.walk(self.dir):
            for filename in filenames:
                self.classes.append(dirname.split("/")[-1])
                self.paths.append(os.path.join(dirname,filename))
        return self.classes, self.paths

# def data_reader(dir):
#     # classes = []
#     # paths = []
#     for dirname, _, filenames in os.walk(dir):
#         for filename


if __name__ == '__main__':
    
    dir = '../datasets/'
    reader = DataReader(dir)
    classes,paths = reader.to_category()
    print(classes)
    # print(classes)
    # print(paths)