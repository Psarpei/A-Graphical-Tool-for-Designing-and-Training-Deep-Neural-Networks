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

class Network_Thread(QtCore.QThread):
    """Thread for training and testing network"""

    signal = QtCore.pyqtSignal('PyQt_PyObject')
    signal2 = QtCore.pyqtSignal('PyQt_PyObject')

    def __init__(self):
        QtCore.QThread.__init__(self)
        self.model = "Network Object"
        self.optimizer = "optimizer"
        self.criterion = "loss function"
        self.epochs = 0
        self.use_gpu = False
        self.train_data = "Train data"
        self.test_data = "Test data"
        self.batch_size = "batch_size"
        self.save_path = "weights folder"
        self.path_weights = "path of current weights"
        self.categories = "list of categories for prediction"

    def set_optimizer(self,type,lr,mom):
        """set optimizer from type with lerning rate lr and momentum mom"""

        if (type == "SGD"):
            #print("set SGD")
            #print(self.model)
            if(mom != ""):
                self.optimizer = optim.SGD(self.model.parameters(), lr=float(lr), momentum=float(mom)) #self.model.parameters(), lr=0.1, momentum=0.8)
            else:
                self.optimizer = optim.SGD(self.model.parameters(), lr=float(lr))
            #print("sgd gesettet")
        elif(type == "Adam"):
            if (lr != ""):
                self.optimizer = optim.Adam(self.model.parameters(), lr=float(lr))

    def set_criterion(self,type):
        """set criterion of type type"""

        if (type == "negativ log likelihood"):
            self.criterion = F.nll_loss
        elif(type == "binary cross entropy"):
            self.criterion = F.binary_cross_entropy

    def train(self, epoch):
        """train the neural network"""
        if(torch.cuda.is_available() and self.use_gpu):
            self.model.cuda()
        self.model.train()
        batch_id = 0
        for data, target in self.train_data:
            if (torch.cuda.is_available() and self.use_gpu):
                data = data.cuda()
            target = torch.Tensor(target)
            if (torch.cuda.is_available() and self.use_gpu):
                target = target.cuda()
            data = Variable(data)
            target = Variable(target)
            self.optimizer.zero_grad()
            out = self.model(data)
            criterion = self.criterion
            #criterion = F.binary_cross_entropy
            loss = criterion(out, target)
            loss.backward()
            self.optimizer.step()
            self.signal.emit('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(data), len(self.train_data) * self.batch_size,
                       100. * batch_id / len(self.train_data), loss.data))
            
            batch_id += + 1

        self.path_weights = self.save_path + "/weights_" +"epoch_"+ str(epoch) + '.pt'
        torch.save(self.model, self.path_weights)

        self.signal2.emit(self.path_weights)

    def test(self):
        """test the neural network"""

        self.model.eval()
        loss = 0
        correct = 0
        incorrect = 0
        preds = 0
        for data, target in self.test_data:
            if (torch.cuda.is_available() and self.use_gpu):
                data = data.cuda()
            data = Variable(data, volatile=True)  # volatile bedeutet kann sich auf der grafikkarte verändern^^
            target = torch.Tensor(target)
            if (torch.cuda.is_available() and self.use_gpu):
                target = target.cuda()
            target = Variable(target)
            out = self.model(data)
            # loss += F.nll_loss(out, target, size_average=False).data
            loss += F.binary_cross_entropy(out, target, size_average=False).data##########################################################################
            prediction = out.data.max(1, keepdim=True)[1]  # out.data -> 64x10
            # print(prediction)
            # print(out.data.size())
            # correct += prediction.eq(target.data.view_as(prediction)).cpu().sum() funktioniert irgendwie nicht xD
            preds += len(prediction)
            for i in range(len(prediction)):
                correct += 1 if (target[i][prediction[i].item()].item() == 1) else 0
                incorrect += 0 if (target[i][prediction[i].item()].item() == 1) else 1
                # print(target[i], target[i][prediction[i].item()].item(), prediction[i].item())

        # print(len(test_data))
        # print(preds)
        #print(correct)
        #print(incorrect)
        loss = loss / (len(self.test_data) * self.batch_size)
        self.signal.emit('Average loss: ' + str(round(loss.item(), 6)))
        #print('Average loss: ', round(loss.item(), 6))
        self.signal.emit("Genauigkeit: " + str(round(100 * correct / (len(self.test_data) * self.batch_size), 2)) + " %")
        #print("Genauigkeit: ", round(100 * correct / (len(self.test_data) * self.batch_size), 2), "%")

    def run(self):
        """start thread next to mainthread and train/test neural network """
        for epoch in range(self.epochs):
            self.train(epoch)
            self.test()
