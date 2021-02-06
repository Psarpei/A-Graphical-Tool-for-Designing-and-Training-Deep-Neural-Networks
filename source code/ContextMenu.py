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

class Context_Menu(QtWidgets.QMenu):
    """Create Context Menu  to edit layer"""

    def __init__(self,layer_num, editable):
        super().__init__()
        self.layer_num = layer_num

        self.action_edit = QtWidgets.QAction(self)
        self.action_edit.setText("Edit")
        self.action_draw = QtWidgets.QAction(self)
        self.action_draw.setText("Draw")
        self.action_delete = QtWidgets.QAction(self)
        self.action_delete.setText("Delete")
        self.action_delete_links = QtWidgets.QAction(self)
        self.action_delete_links.setText("Delete Links")
        self.action_first_layer = QtWidgets.QAction(self)
        self.action_first_layer.setText("First Layer")

        if(editable):
            self.addAction(self.action_edit)
        self.addAction(self.action_draw)
        self.addAction(self.action_delete)
        self.addAction(self.action_delete_links)
        self.addAction(self.action_first_layer)
