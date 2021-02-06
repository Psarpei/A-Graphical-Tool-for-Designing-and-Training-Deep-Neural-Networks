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
class Network(nn.Module):
    """Creating Neural Network with Pytorch"""

    def __init__(self,layer_list):
        super(Network,self).__init__()
        self.functions = []
        layers = []
        for layer in layer_list:
            self.functions.append(self.create_layer(layer))

        for layer in self.functions:
            if(not isinstance(layer, list)):
                #print(layer)
                #print(isinstance(layer, list))
                layers.append(layer)

        self.layer = nn.ModuleList(layers)
        #print("moduleList  created")

    def forward(self, x):
        """ return prediction from network for x"""
        layer_num = 0
        for layer in self.functions:
            if(isinstance(layer, list)):
                if(layer[0] == "ReLu"):
                    #print("Relu")
                    x = F.relu(x)
                elif(layer[0] == "Sigmoid"):
                    #print("sigmoid")
                    x = torch.sigmoid(x)
                elif(layer[0] == "Max Pool"):
                    #print("max pool")
                    x = F.max_pool2d(x, layer[1])
                    #print("type",type(layer[1]))
                elif(layer[0] == "Avg Pool"):
                    #print("avg pool")
                    x = F.avg_pool2d(x, layer[1])
                    #print("type", type(layer[1]))

                elif(layer[0] == "View"):
                    #print("view")
                    #print(x.size())
                    #print(layer[1])
                    x = x.view(-1,layer[1])
                    #print("type", type(layer[1]))
                else:
                    pass
                    #print("SOMETHIG IS WRONG???????")
            else:
                #print("else")
                #print(self.layer[layer_num])
                #print(x.size())
                x = self.layer[layer_num](x)
                layer_num += 1
        return x

    def create_layer(self, layer_tupel):
        """create layer from layer_tupel of world object"""
        if(layer_tupel[0] == "Linear Layer"):
            return nn.Linear(layer_tupel[1],layer_tupel[2])
            #print("linear type", layer_tupel[2], type[layer_tupel[2]])
        elif(layer_tupel[0] == "Convolution Layer"):
            return nn.Conv2d(layer_tupel[1],layer_tupel[2],kernel_size=layer_tupel[3])
            #print("conv type",layer_tupel[2], type(layer_tupel[2]), type(layer_tupel[3]))
        elif(layer_tupel[0] == "Dropout"):
            return nn.Dropout2d()
        elif(layer_tupel[0] in ["View","Avg Pool", "Max Pool"]):
            return [layer_tupel[0],layer_tupel[1]]
        else:
            return [layer_tupel[0]]
