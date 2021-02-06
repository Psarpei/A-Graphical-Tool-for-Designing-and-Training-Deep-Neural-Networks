from PyQt5 import QtCore, QtGui, QtWidgets
from WorldWidget import *
from ContextMenu import *
from DataDialog import *
from DataThread import *
from Layer import *
from LayerDialog import *
from MainWindow import *
from NetworkThread import *
from Networkx import *
from NeuralNetwork import *
from TestPictureDialog import *
from WorldWidget import *

import torch
import torchvision
from torchvision import transforms
from PIL import Image
from os import listdir
import random
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import random
import time
import os
import sys

class Data_Thread(QtCore.QThread):
    """Thread for creating train and test data """
    signal = QtCore.pyqtSignal('PyQt_PyObject')

    def __init__(self):
        QtCore.QThread.__init__(self)
        self.path_train = ""
        self.path_test = ""
        self.train_data = []
        self.test_data = []
        self.batch_size = 0
        self.resize = 0
        self.transforms = 0
        self.normalize = 0
        self.type = 0

    def __del__(self):
        self.wait()

    def picture1(self):
        """create train and test data from pictures of type 1"""

        for path in [self.path_train, self.path_test]:
            # path = self.path_train
            batch_size = self.batch_size
            transforms = self.transforms
            data = []
            files = []
            data_list = []
            target_list = []
            categories_for_choice = listdir(path)
            categories_for_index = listdir(path)
            category_size = len(categories_for_choice)
            file_number = 0

            for cat in listdir(path):
                files.append(listdir(path + "/" + cat))
                file_number += len(listdir(path + "/" + cat))

            for i in range(file_number):
                cat_index = random.randint(0, len(categories_for_index) - 1)
                file = random.choice(files[cat_index])
                img = Image.open(path + "/" + categories_for_index[cat_index] + "/" + file)
                img_tensor = transforms(img)
                data_list.append(img_tensor)
                target = [0] * category_size
                target[categories_for_choice.index(categories_for_index[cat_index])] = 1
                target_list.append(target)

                if (len(data_list) >= batch_size):
                    data.append((torch.stack(data_list), target_list))
                    data_list = []
                    target_list = []
                    self.signal.emit('Loaded batch ' + str(len(data)) + ' of ' + str(file_number // batch_size))
                    self.signal.emit('Percentage Done: ' + str(round(100 * len(data) / (file_number // batch_size), 2)) + ' %')
                    # self.signal.emit(str(i))

                    #print('Loaded batch ', len(data), 'of ', file_number // batch_size)
                    #print('Percentage Done: ', round(100 * len(data) / (file_number // batch_size), 2), '%')
                files[cat_index].remove(file)
                if (len(files[cat_index]) == 0):
                    files.pop(cat_index)
                    categories_for_index.pop(cat_index)

                if (path == self.path_train):
                    self.train_data = data
                else:
                    self.test_data = data

                if(self.path_train == self.path_test):
                    self.test_data = data

                self.categories = listdir(self.path_train)

    def picture2(self):
        """create train and test data from pictures of type 2"""

        #print(1)
        for path in [self.path_train, self.path_test]:
            batch_size = self.batch_size
            transforms = self.transforms
            data = []
            data_list = []
            target_list = []
            files = listdir(path)
            #print(2)
            for i in range(len(listdir(path))):
                #print(2.1)
                f = random.choice(files)
                #print(2.2)
                files.remove(f)
                #print(2.3)
                #print(path+f)
                img = Image.open(path + "/" + f)
                #print(2.4)
                img_tensor = transforms(img)  # (3,256,256)
                #print(2.5)
                data_list.append(img_tensor)
                #print(3)
                target = []
                #print(f)
                for cat in self.categories:
                    if cat in f:
                        target.append(1)
                    else:
                        target.append(0)
                #print(target)
                target_list.append(target)
                #print(4)
                if len(data_list) >= batch_size:
                    data.append((torch.stack(data_list), target_list))
                    data_list = []
                    target_list = []
                    #print(5)
                    self.signal.emit('Loaded batch ' + str(len(data)) + ' of ' + str(int(len(listdir(path)) / batch_size)))
                    #print('Loaded batch ', len(data), 'of ', int(len(listdir(path)) / batch_size))
                    #print(6)
                    self.signal.emit('Percentage Done: ' + str(round(100 * len(data) / int(len(listdir(path)) / batch_size),2)) + '%')
                    #print('Percentage Done: ', 100 * len(data) / int(len(listdir(path)) / batch_size), '%')
                    #print(7)

            if (path == self.path_train):
                self.train_data = data
            else:
                self.test_data = data

            if (self.path_train == self.path_test):
                self.test_data = data

            #self.categories = ["Cat","Dog"]

    def run(self):
        "start creating train/test data in a thread next to the mainthread"

        if (self.type == "Picture1"):
            self.picture1()
        elif (self.type == "Picture2"):
            self.picture2()
