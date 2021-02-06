from PyQt5 import QtCore, QtGui, QtWidgets
from WorldWidget import *
from ContextMenu import *
from DataDialog import *
from DataThread import *
from Layer import *
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

class World(QtWidgets.QWidget):
    """ Widget for  creating grafical neural network """

    signal = QtCore.pyqtSignal('PyQt_PyObject')

    def __init__(self, layer_list, connection_list, first_layer):
        super().__init__()
        #print(first_layer)
        self.network_layers = layer_list
        self.connections = connection_list
        self.first_layer = first_layer
        self.mouse_cached = -1
        self.mouse_pos = 0
        self.draw_from = -1
        self.mouse_x = 0
        self.mouse_y = 0
        self.change_layer = 0
        self.setMouseTracking(True)
        self.dialog = 0
        self.context_menu = 0
        self.init_world()

    def init_world(self):
        self.setGeometry(QtCore.QRect(0, 0, 50000, 50000))
        self.setMinimumSize(QtCore.QSize(50000, 50000))
        self.setObjectName("world")
        for layer in self.network_layers:
            layer.setParent(self)

        #if(self.first_layer != -1):
        #    self.network_layers[self.first_layer].widget.setStyleSheet("background-color: rgb(0, 204, 51);\n"
        #                          "border: 1px solid black;")

        #self.add_layer("Linear")

    def get_layer(self,layer_num):
        """return tupel with important information about the layer_num layer"""
        if(self.network_layers[layer_num].type == "Linear Layer"):
            return (self.network_layers[layer_num].type,self.network_layers[layer_num].input_size, self.network_layers[layer_num].output_size)
        elif (self.network_layers[layer_num].type == "Convolution Layer"):
            return (self.network_layers[layer_num].type,self.network_layers[layer_num].input_size, self.network_layers[layer_num].output_size, self.network_layers[layer_num].kernel_size)
        elif (self.network_layers[layer_num].type in ["Max Pool", "Avg Pool"]):
            return (self.network_layers[layer_num].type,self.network_layers[layer_num].kernel_size)
        elif (self.network_layers[layer_num].type == "View"):
            return (self.network_layers[layer_num].type, self.network_layers[layer_num].input_size)
        else:
            return(self.network_layers[layer_num].type,0)

    def get_network(self):
        """create list with important informations about the neural network from network_layers and connections"""
        actual = self.first_layer
        network = []
        has_follower = True
        while(has_follower):
            has_follower = False
            network.append(self.get_layer(actual))
            for link in self.connections:
                if(link[0] == actual):
                    actual = link[1]
                    has_follower = True
                    break
        #print(network)
        return network

    def create_list_for_save(self):
        """return list of layersettings and geometrical information for save grafical network"""
        #print("creeeeeeeate")
        save_list = []
        for i in range(len(self.network_layers)):
            settings = self.get_layer(i)
            geometrie = (self.network_layers[i].x() ,self.network_layers[i].y())
            save_list.append((settings,geometrie))
        return save_list

    def set_layer_settings(self, connections, load_list, first_layer):
        """set layer settings by getting information from a loaded network"""
        self.connections = connections
        self.set_first_layer(first_layer)
        for i in range(len(self.network_layers)):
            if (self.network_layers[i].type == "Linear Layer"):
                self.network_layers[i].input_size = load_list[i][0][1]
                self.network_layers[i].output_size = load_list[i][0][2]
            elif (self.network_layers[i].type == "Convolution Layer"):
                self.network_layers[i].input_size = load_list[i][0][1]
                self.network_layers[i].output_size = load_list[i][0][2]
                self.network_layers[i].kernel_size = load_list[i][0][3]
            elif (self.network_layers[i].type in ["Max Pool", "Avg Pool"]):
                self.network_layers[i].kernel_size = load_list[i][0][1]
            elif (self.network_layers[i].type == "View"):
                self.network_layers[i].input_size = load_list[i][0][1]
            self.network_layers[i].move(load_list[i][1][0],load_list[i][1][1])
            if(not isinstance(self.network_layers[i], Function_Layer)):
                self.network_layers[i].update_settings()

    def add_layer(self,layer):
        """create a new layer"""
        if (layer == "Linear Layer"):
            self.network_layers.append(Linear_Layer(self))
        elif(layer == "Convolution Layer"):
            self.network_layers.append(Convolution_Layer(self))
        elif(layer in ["Max Pool","Avg Pool"]):
            self.network_layers.append(Pooling_Layer(self,layer))
        elif(layer == "View"):
            self.network_layers.append(View_Layer(self))
        else:
            self.network_layers.append(Function_Layer(self, layer))
        self.network_layers[-1].move(50,50)

        if(self.first_layer == -1):
            self.first_layer = len(self.network_layers)-1
            #print("jaa?", self.first_layer)
            self.network_layers[len(self.network_layers)-1].widget.setStyleSheet("background-color: rgb(0, 204, 51);\n"
                                  "border: 1px solid black;")
            self.update()


    def mousePressEvent(self, event):
        if(event.button() == 1):
            for i in range(len(self.network_layers)):
                if(self.is_overlapping(i, event)):
                    self.draw_to_layer(i) #for setting connection
                    self.mouse_cached = i
                    self.mouse_pos = event.pos()
                    self.setCursor(QtCore.Qt.ClosedHandCursor)
                    break

        elif(event.button() == 2):#create context menu for layer which overlapps point of right mouseclick
            for i in range(len(self.network_layers)):
                if self.is_overlapping(i, event):
                    self.create_context_menu(event, i)
                    break

    def create_connection(self, layer_num):
        """create connection start or and"""
        if (self.draw_from != -1):
            self.draw_to_layer(layer_num)
        else:
            self.draw_from_layer(layer_num)

    def draw_from_layer(self,layer_num):
        """set layer_num as start from new connections"""
        if (self.check_connections(layer_num, 0)):
            self.draw_from = layer_num

    def draw_to_layer(self, layer_num):
        """create new network connection from actual draw_from to layer_num"""

        if (not self.draw_from in [layer_num,-1]):
            if (self.check_connections(layer_num, 1)):
                self.connections.append((self.draw_from, layer_num))
        self.draw_from = -1
        self.update()

    def create_context_menu(self,event,layer_num):
        """create context menu for right clicked layer"""

        editable = not isinstance(self.network_layers[layer_num], Function_Layer)
        self.context_menu = Context_Menu(layer_num, editable)
        if(editable):
            self.context_menu.action_edit.triggered.connect(lambda: self.open_layer_dialog(layer_num))
        self.context_menu.action_draw.triggered.connect(lambda: self.draw_from_layer(layer_num))
        self.context_menu.action_delete.triggered.connect(lambda: self.delete_layer(layer_num))
        self.context_menu.action_delete_links.triggered.connect(lambda: self.delete_links(layer_num))
        self.context_menu.action_first_layer.triggered.connect(lambda: self.set_first_layer(layer_num))
        action = self.context_menu.exec_(self.mapToGlobal(event.pos()))

    def set_first_layer(self, layer_num):
        """set layer_num layer as first_layer and change background-color of current/last first_layer """
        self.network_layers[self.first_layer].widget.setStyleSheet("background-color: rgb(211, 211, 211);\n"
                                  "border: 1px solid black;")

        self.first_layer = layer_num
        self.network_layers[layer_num].widget.setStyleSheet("background-color: rgb(0, 204, 51);\n"
                                  "border: 1px solid black;")

        for link in self.connections:
            if(link[1] == layer_num):
                self.connections.remove(link)


    def mouseReleaseEvent(self, event):
        """if left mouse button released cached layer becomes free and mouse becomes to OpenHandCursor"""
        self.mouse_cached = -1
        #self.setCursor(QtCore.Qt.ArrowCursor)
        self.setCursor(QtCore.Qt.OpenHandCursor)

    def mouseMoveEvent(self, event):
        """handle movement of drawing connection or move layer"""
        if(self.mouse_cached != -1): #move layer
            movement = event.pos() - self.mouse_pos
            self.mouse_pos = event.pos()
            self.network_layers[self.mouse_cached].move(self.network_layers[self.mouse_cached].pos() + movement)
            self.update()
        elif(self.draw_from != -1): #draw connection
            self.mouse_x = event.x()
            self.mouse_y = event.y()
            self.update()
        else: #reset cursor to OpenHandCursor
            self.setCursor(QtCore.Qt.OpenHandCursor)

    def mouseDoubleClickEvent(self, event):
        """double click on layer activate """
        for i in range(len(self.network_layers)):
            if (self.is_overlapping(i, event)):
                self.create_connection(i)
                break

    def open_layer_dialog(self, i):
        """open layer dialog for edit layer settings like input/output size or something like that """
        self.change_layer = i
        self.dialog = Layer_Dialog(self.network_layers[i].type)
        self.dialog.buttonBox.accepted.connect(self.layer_change)

    def paintEvent(self, event):
        """paintEvent only for painting connections"""
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QColor(0,0,0))
        painter.setFont(QtGui.QFont("Arial", 400))
        if(self.draw_from != -1):
            painter.drawLine(self.network_layers[self.draw_from].x()+self.network_layers[self.draw_from].width()/
                             2,self.network_layers[self.draw_from].y()+self.network_layers[self.draw_from].height(),
                             self.mouse_x, self.mouse_y)
        for link in self.connections:
            painter.drawLine(self.network_layers[link[0]].x()+self.network_layers[link[0]].width()/2,
                             self.network_layers[link[0]].y()+self.network_layers[link[0]].height(),
                             self.network_layers[link[1]].x()+self.network_layers[link[1]].width()/2,
                             self.network_layers[link[1]].y())

    def is_overlapping(self, layer_num, event):
        if (event.x() >= self.network_layers[layer_num].x() - 5 and event.x() - 5 <= self.network_layers[layer_num].x()
                + self.network_layers[layer_num].width() and event.y() >= self.network_layers[layer_num].y() - 5
                and event.y() - 5 <= self.network_layers[layer_num].y() + self.network_layers[layer_num].height()):
            return True
        else:
            return False

    def check_connections(self,layer_num,index):
        """check if connection is possible and return True if it's possible else False"""
        if(layer_num == self.first_layer and index == 1):
            return False
        for link in self.connections:
            if(link[index] == layer_num):
                return False
        return True

    def layer_change(self):
        """change layer from new settings of the layer_dialog from before"""
        if(self.network_layers[self.change_layer].type not in ["Max Pool", "Avg Pool"]):
            #print("neeeeeeeeeeeein?")
            self.network_layers[self.change_layer].input_size = int(self.dialog.input_spinBox.value())
            if(self.network_layers[self.change_layer].type != "View"):
                self.network_layers[self.change_layer].output_size = int(self.dialog.output_spinBox.value())
        if(self.network_layers[self.change_layer].type in ["Convolution Layer","Max Pool", "Avg Pool"]):
            #print(1)
            self.network_layers[self.change_layer].kernel_size = int(self.dialog.kernel_spinBox.value())
            #print(2)
        #self.signal.emit(" ")
        self.network_layers[self.change_layer].update_settings()
        self.dialog.reject()
        self.update()

    def delete_links(self, layer_num):
        """delete input and output link from layer_num layer"""
        to_remove = []
        for i in range(len(self.connections)):
            if layer_num in self.connections[i]:
                to_remove.append(i - len(to_remove))
        for index in to_remove:
            self.connections.pop(index)

    def delete_layer(self,layer_num):
        """delete layer_num layer"""
        self.network_layers[layer_num].deleteLater()
        self.network_layers.pop(layer_num)

        self.delete_links(layer_num)

        new_connections = []
        for link in self.connections:
            zero = link[0] if(link[0] < layer_num) else link[0]-1
            one = link[1] if(link[1] < layer_num) else link[1]-1
            new_connections.append((zero,one))
        if(self.first_layer >= layer_num):
            self.first_layer = -1 if(self.first_layer == layer_num) else self.first_layer -1
        self.connections = new_connections
