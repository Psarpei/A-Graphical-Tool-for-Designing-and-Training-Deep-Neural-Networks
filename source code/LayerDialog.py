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
class Layer_Dialog(QtWidgets.QDialog):
    """Dialog for change layer settings"""

    def __init__(self,layer_type):
        super().__init__()
        self.resize(400, 279)
        self.setWindowTitle("Change layer")
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

        if(layer_type in ["Max Pool", "Avg Pool"]):
            #print("1")
            self.kernel_label = QtWidgets.QLabel(self.gridLayoutWidget)
            font = QtGui.QFont()
            font.setPointSize(14)
            #print("2")
            self.kernel_label.setFont(font)
            self.kernel_label.setText("Kernel size:")
            self.kernel_label.setAlignment(QtCore.Qt.AlignCenter)
            #print("3")
            self.gridLayout.addWidget(self.kernel_label, 0, 0, 1, 1)
            self.kernel_spinBox = QtWidgets.QSpinBox(self.gridLayoutWidget)
            self.kernel_spinBox.setMinimum(1)
            self.kernel_spinBox.setMaximum(9999999)
            #print("4")
            font = QtGui.QFont()
            font.setPointSize(12)
            self.kernel_spinBox.setFont(font)
            self.gridLayout.addWidget(self.kernel_spinBox, 0, 1, 1, 1)
            #print("5")
        else:
            self.input_label = QtWidgets.QLabel(self.gridLayoutWidget)
            font = QtGui.QFont()
            font.setPointSize(14)
            self.input_label.setFont(font)
            self.input_label.setText("input size:")
            self.input_label.setAlignment(QtCore.Qt.AlignCenter)
            self.gridLayout.addWidget(self.input_label, 0, 0, 1, 1)
            self.input_spinBox = QtWidgets.QSpinBox(self.gridLayoutWidget)
            self.input_spinBox.setMinimum(1)
            self.input_spinBox.setMaximum(9999999)
            font = QtGui.QFont()
            font.setPointSize(12)
            self.input_spinBox.setFont(font)
            self.gridLayout.addWidget(self.input_spinBox, 0, 1, 1, 1)

            if(layer_type != "View"):
                self.output_label = QtWidgets.QLabel(self.gridLayoutWidget)
                font = QtGui.QFont()
                font.setPointSize(14)
                self.output_label.setFont(font)
                self.output_label.setText("output size:")
                self.gridLayout.addWidget(self.output_label, 1, 0, 1, 1, QtCore.Qt.AlignHCenter)
                self.output_spinBox = QtWidgets.QSpinBox(self.gridLayoutWidget)
                self.output_spinBox.setMinimum(1)
                self.output_spinBox.setMaximum(9999999)
                font = QtGui.QFont()
                font.setPointSize(12)
                self.output_spinBox.setFont(font)
                self.gridLayout.addWidget(self.output_spinBox, 1, 1, 1, 1)

            if (layer_type == "Convolution Layer"):
                font = QtGui.QFont()
                font.setPointSize(14)
                self.kernel_label = QtWidgets.QLabel(self.gridLayoutWidget)
                self.kernel_label.setFont(font)
                self.kernel_label.setText("kernel size:")
                self.kernel_label.setAlignment(QtCore.Qt.AlignCenter)
                self.gridLayout.addWidget(self.kernel_label, 2, 0, 1, 1)
                self.kernel_spinBox = QtWidgets.QSpinBox(self.gridLayoutWidget)
                font = QtGui.QFont()
                font.setPointSize(12)
                self.kernel_spinBox.setFont(font)
                self.kernel_spinBox.setMaximum(9999999)
                self.kernel_spinBox.setMinimum(1)
                self.gridLayout.addWidget(self.kernel_spinBox, 2, 1, 1, 1)

        #self.retranslateUi(CreateLinearLayer)
        #self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.show()
