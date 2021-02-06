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

class Layer(QtWidgets.QWidget):
    """basis layer widget"""

    def __init__(self,parent):
        super().__init__(parent)
        self.init_layer()

    def init_layer(self):
        self.widget = QtWidgets.QWidget(self)
        self.setMinimumSize(300,100)
        #self.widget.setGeometry(QtCore.QRect(180, 170, 321, 101))
        self.widget.setStyleSheet("background-color: rgb(211, 211, 211);\n"
                                  "border: 1px solid black;")
        self.widget.setObjectName("widget")
        self.widget.setMinimumSize(300, 100)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.name_label = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.name_label.setFont(font)
        self.name_label.setText("Convolution Layer")
        self.verticalLayout.addWidget(self.name_label, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self.settings_label = QtWidgets.QLabel(self.widget)
        self.settings_label.setText("in: 10000 out: 30000 kernel: 500x500")
        font = QtGui.QFont()
        font.setPointSize(10)
        self.settings_label.setFont(font)
        self.verticalLayout.addWidget(self.settings_label, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

class Linear_Layer(Layer):
    """Linear layer widget"""

    def __init__(self,parent):
        super().__init__(parent)
        self.init_layer()
        self.type = "Linear Layer"
        self.input_size = 50
        self.output_size = 200
        self.name_label.setText(self.type)
        self.update_settings()

    def update_settings(self):
        """update settings text for layer in GUI"""

        self.settings_label.setText("in: "+ str(self.input_size) + " out: " + str(self.output_size))

class Convolution_Layer(Layer):
    """Convolution layer widget"""

    def __init__(self,parent):
        super().__init__(parent)
        self.init_layer()
        self.type = "Convolution Layer"
        self.input_size = 50
        self.output_size = 200
        self.kernel_size = 50
        self.name_label.setText(self.type)
        self.update_settings()

    def update_settings(self):
        """update settings text for layer in GUI"""
        self.settings_label.setText("in: "+ str(self.input_size) + " out: " + str(self.output_size) +
                                    " kernel: " + str(self.kernel_size) + "x" + str(self.kernel_size))

class Pooling_Layer(Layer):
    """Pooling layer widget"""
    def __init__(self,parent,type):
        super().__init__(parent)
        self.init_layer()
        self.type = type
        self.kernel_size = 5
        self.name_label.setText(self.type)
        self.update_settings()

    def update_settings(self):
        """update settings text for layer in GUI"""
        self.settings_label.setText("kernel: " + str(self.kernel_size) + "x" + str(self.kernel_size))

class View_Layer(Layer):
    """View layer widget"""
    def __init__(self,parent):
        super().__init__(parent)
        self.init_layer()
        self.type = "View"
        self.input_size = 100
        self.name_label.setText(self.type)
        self.update_settings()

    def update_settings(self):
        """update settings text for layer in GUI"""
        self.settings_label.setText("in/out: " + str(self.input_size))


class Function_Layer(QtWidgets.QWidget):
    def __init__(self,parent,type):
        super().__init__(parent)
        self.type = type
        self.init_layer()

    def init_layer(self):
        self.setMinimumSize(200, 60)
        self.widget = QtWidgets.QWidget(self)
        self.widget.setStyleSheet("background-color: rgb(211, 211, 211);\n"
                                  "border: 1px solid black;")
        self.widget.setMinimumSize(200,60)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.name_label = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.name_label.setFont(font)
        self.name_label.setAlignment(QtCore.Qt.AlignCenter)
        self.name_label.setText(self.type)
        self.verticalLayout.addWidget(self.name_label)
