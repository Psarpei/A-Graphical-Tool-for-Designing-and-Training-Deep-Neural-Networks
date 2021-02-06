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

class Train_Dialog(QtWidgets.QDialog):
    """Dialog for train neural network"""

    def __init__(self):
        super().__init__()
        self.resize(400, 279)
        self.setWindowTitle("Train Settings")
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
        self.loss_label = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.loss_label.setFont(font)
        self.loss_label.setText("loss function:")
        self.gridLayout.addWidget(self.loss_label, 1, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.optimizer_label = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.optimizer_label.setFont(font)
        self.optimizer_label.setText("optimizer:")
        self.optimizer_label.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(self.optimizer_label, 0, 0, 1, 1)
        self.optimizer_comboBox = QtWidgets.QComboBox(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.optimizer_comboBox.setFont(font)
        self.optimizer_comboBox.addItem("SGD")
        self.optimizer_comboBox.addItem("Adam")
        self.gridLayout.addWidget(self.optimizer_comboBox, 0, 1, 1, 1)
        self.loss_comboBox = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.loss_comboBox.addItem("negativ log likelihood")
        self.loss_comboBox.addItem("binary cross entropy")
        font = QtGui.QFont()
        font.setPointSize(12)
        self.loss_comboBox.setFont(font)
        self.gridLayout.addWidget(self.loss_comboBox, 1, 1, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.epochs_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.epochs_label.setFont(font)
        self.epochs_label.setText("epochs:")
        self.epochs_label.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(self.epochs_label, 2, 0, 1, 1)
        self.epochs_spinBox= QtWidgets.QSpinBox(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.epochs_spinBox.setFont(font)
        self.gridLayout.addWidget(self.epochs_spinBox, 2, 1, 1, 1)

        self.lr_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.lr_label.setFont(font)
        self.lr_label.setText("Learning rate:")
        self.lr_label.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(self.lr_label, 3, 0, 1, 1)
        self.lr_line_edit = QtWidgets.QLineEdit(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lr_line_edit.setFont(font)
        self.gridLayout.addWidget(self.lr_line_edit, 3, 1, 1, 1)

        self.momentum_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.momentum_label.setFont(font)
        self.momentum_label.setText("momentum:")
        self.momentum_label.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(self.momentum_label, 4, 0, 1, 1)
        self.momentum_line_edit = QtWidgets.QLineEdit(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.momentum_line_edit.setFont(font)
        self.gridLayout.addWidget(self.momentum_line_edit , 4, 1, 1, 1)

        font = QtGui.QFont()
        font.setPointSize(14)
        self.gpu_radio_Button = QtWidgets.QRadioButton(self.gridLayoutWidget)
        self.gpu_radio_Button.setFont(font)
        self.gpu_radio_Button.setText("GPU")
        #self.gpu_radio_Button.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(self.gpu_radio_Button, 5, 1, 1, 1)

        #self.retranslateUi(CreateLinearLayer)
        #self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.show()
