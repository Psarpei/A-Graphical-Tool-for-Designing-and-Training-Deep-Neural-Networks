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
class Data_Dialog(QtWidgets.QDialog):
    """Dialog for Creating train/test data"""

    def __init__(self, has_batch, cat_to_set):
        super().__init__()
        self.resize(400, 279)
        self.setWindowTitle("Load Data")
        self.buttonBox = QtWidgets.QDialogButtonBox(self)
        self.buttonBox.setGeometry(QtCore.QRect(30, 240, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayoutWidget = QtWidgets.QWidget(self)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(9, 9, 391, 211))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.resize_label = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.resize_label.setFont(font)
        self.resize_label.setText("Resize:")
        self.resize_label.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(self.resize_label, 0, 0, 1, 1)
        self.resize_spinBox = QtWidgets.QSpinBox(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.resize_spinBox.setFont(font)
        self.resize_spinBox.setMinimum(1)
        self.resize_spinBox.setMaximum(9999999)
        self.gridLayout.addWidget(self.resize_spinBox, 0, 1, 1, 1)
        self.batch_label = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        if(has_batch):
            self.batch_label.setFont(font)
            self.batch_label.setText("batch size:")
            self.gridLayout.addWidget(self.batch_label, 1, 0, 1, 1, QtCore.Qt.AlignHCenter)
            self.batch_spinBox = QtWidgets.QSpinBox(self.gridLayoutWidget)
            self.batch_spinBox.setMinimum(1)
            self.batch_spinBox.setMaximum(9999999)
            font = QtGui.QFont()
            font.setPointSize(12)
            self.batch_spinBox.setFont(font)
            self.gridLayout.addWidget(self.batch_spinBox, 1, 1, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.norm_mean_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.norm_mean_label.setFont(font)
        self.norm_mean_label.setText("normalize mean:")
        self.norm_mean_label.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(self.norm_mean_label, 2, 0, 1, 1)
        self.norm_mean_input= QtWidgets.QLineEdit(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.norm_mean_input.setFont(font)
        self.gridLayout.addWidget(self.norm_mean_input, 2, 1, 1, 1)

        font = QtGui.QFont()
        font.setPointSize(14)
        self.norm_std_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.norm_std_label.setFont(font)
        self.norm_std_label.setText("normalize std:")
        self.norm_std_label.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(self.norm_std_label, 3, 0, 1, 1)
        self.norm_std_input = QtWidgets.QLineEdit(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.norm_std_input.setFont(font)
        self.gridLayout.addWidget(self.norm_std_input, 3, 1, 1, 1)

        if(cat_to_set):
            font = QtGui.QFont()
            font.setPointSize(14)
            self.cat_label = QtWidgets.QLabel(self.gridLayoutWidget)
            self.cat_label.setFont(font)
            self.cat_label.setText("categories:")
            self.cat_label.setAlignment(QtCore.Qt.AlignCenter)
            self.gridLayout.addWidget(self.cat_label, 4, 0, 1, 1)
            self.cat_input = QtWidgets.QLineEdit(self.gridLayoutWidget)
            font = QtGui.QFont()
            font.setPointSize(12)
            self.cat_input.setFont(font)
            self.gridLayout.addWidget(self.cat_input, 4, 1, 1, 1)

        #self.retranslateUi(CreateLinearLayer)
        #self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.show()
