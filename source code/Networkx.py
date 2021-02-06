from PyQt5 import QtCore, QtGui, QtWidgets
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

class Console(QtWidgets.QWidget):
    """Console for text information"""

    def __init__(self, text_list):
        super().__init__()
        self.text_list = []
        self.v_layout = QtWidgets.QVBoxLayout(self)
        self.init_console(text_list)
        self.setMouseTracking(True)

    def init_console(self,text_list):
        self.setObjectName("console")
        self.v_layout.setObjectName("console_Vlayout")

        for text in text_list:##############################################################################
            self.add_text(text)

    def add_text(self, text):
        """add text to console"""

        self.text_list.append(QtWidgets.QLabel())
        self.text_list[-1].setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        #self.text[-1].setObjectName("console_start_text")
        self.text_list[-1].setText(text)
        self.text_list[-1].setMinimumSize(QtCore.QSize(1800, 20))
        self.v_layout.addWidget(self.text_list[-1])

    def mouseMoveEvent(self, event):
        """change cursor to ArrowCursor"""

        self.setCursor(QtCore.Qt.ArrowCursor)


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

class Test_Picture_Dialog(QtWidgets.QDialog):
    """dialog for testing neural network for on picture prediction"""

    def __init__(self):
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
        self.batch_label = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.batch_label.setFont(font)
        self.batch_label.setText("batch size:")
        self.gridLayout.addWidget(self.batch_label, 1, 0, 1, 1, QtCore.Qt.AlignHCenter)
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
        self.gridLayout.addWidget(self.resize_spinBox, 0, 1, 1, 1)
        self.batch_spinBox = QtWidgets.QSpinBox(self.gridLayoutWidget)
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

        #self.retranslateUi(CreateLinearLayer)
        #self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.show()

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

class Test_Picture_Dialog(QtWidgets.QDialog):
    """Dialog for show picture and prediction of picture"""

    def __init__(self,path, pred):
        super().__init__()
        self.resize(400, 300)
        self.buttonBox = QtWidgets.QDialogButtonBox(self)
        self.buttonBox.setGeometry(QtCore.QRect(30, 240, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayoutWidget = QtWidgets.QWidget(self)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(19, 9, 371, 191))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.picture = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.picture.setText("Picture")
        self.pixmap = QtGui.QPixmap(path)
        self.picture.setPixmap(self.pixmap)
        self.verticalLayout.addWidget(self.picture, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.predict = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.predict.setText("Prediction: " + str(pred))
        self.verticalLayout.addWidget(self.predict, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom)

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.show()

class Main_Window(QtWidgets.QMainWindow):
    """main window object"""

    def __init__(self):
        super().__init__()
        self.world = World([],[],-1)
        self.world.add_layer("Linear Layer")
        self.setMouseTracking(True)
        self.mouse_pos = 0
        self.resizing = False
        self.dialog = "Dialog object"
        self.data_thread = Data_Thread()
        self.network_thread = Network_Thread()
        self.setupUi()

    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(921, 816)
        self.window_Hlayout = QtWidgets.QWidget(self)
        self.window_Hlayout.setObjectName("window_Hlayout")
        self.window_Vlayout = QtWidgets.QVBoxLayout(self.window_Hlayout)
        self.window_Vlayout.setObjectName("window_VLayout")
        self.window_Vlayout.setContentsMargins(30,30,30,30)
        self.window_Vlayout.setSpacing(30)

        self.scroll_world = QtWidgets.QScrollArea(self.window_Hlayout)
        self.scroll_world.setStyleSheet("background-color: rgb(255, 255, 255);")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

        self.scroll_world.setObjectName("scroll_world")

        self.scroll_world.setWidget(self.world)
        self.window_Vlayout.addWidget(self.scroll_world)
        #self.window_Vlayout.setContentsMargins(30,30,30,30)

        self.scroll_console = QtWidgets.QScrollArea(self.window_Hlayout)
        self.scroll_console.setMaximumSize(QtCore.QSize(16777215, 100))
        self.scroll_console.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.scroll_console.setObjectName("scroll_console")

        self.console = Console([])
        self.create_text()

        self.scroll_console.setWidget(self.console)
        self.window_Vlayout.addWidget(self.scroll_console)

        self.button = QtWidgets.QPushButton(self.window_Hlayout)
        self.button.setMaximumSize(100,50)
        self.button.setText("clear")
        self.button.clicked.connect(self.clear_console)

        self.window_Vlayout.addWidget(self.button)

        self.window_Hlayout.setMouseTracking(True)
        self.setCentralWidget(self.window_Hlayout)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 921, 26))
        self.menubar.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.menu_file = QtWidgets.QMenu(self.menubar)
        self.menu_file.setTitle("File")
        self.menu_layer = QtWidgets.QMenu(self.menubar)
        self.menu_layer.setTitle("Layer")
        self.menu_functions = QtWidgets.QMenu(self.menubar)
        self.menu_functions.setTitle("Functions")
        self.menu_data = QtWidgets.QMenu(self.menubar)
        self.menu_data.setTitle("Data")
        #menu_train
        self.menu_test = QtWidgets.QMenu(self.menubar)
        self.menu_test.setTitle("Test")

        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        self.action_load_network = QtWidgets.QAction(self)
        self.action_load_network.setText("Load Network")
        self.action_load_network.triggered.connect(self.load_network)

        self.action_save_network = QtWidgets.QAction(self)
        self.action_save_network.setText("Save Network")
        self.action_save_network.triggered.connect(self.save_network)

        self.action_export_py = QtWidgets.QAction(self)
        self.action_export_py.setText("Export .py")
        self.action_export_py.triggered.connect(self.export_network_as_py)

        self.action_conv = QtWidgets.QAction(self)
        self.action_conv.setText("Convolution Layer")
        self.action_conv.triggered.connect(lambda: self.create_layer("Convolution Layer"))

        self.action_linear = QtWidgets.QAction(self)
        self.action_linear.setText("Linear Layer")
        self.action_linear.triggered.connect(lambda: self.create_layer("Linear Layer"))

        self.action_relu = QtWidgets.QAction(self)
        self.action_relu.setText(("ReLu"))
        self.action_relu.triggered.connect(lambda: self.create_layer("ReLu"))

        self.action_sigmoid = QtWidgets.QAction(self)
        self.action_sigmoid.setText("Sigmoid")
        self.action_sigmoid.triggered.connect(lambda: self.create_layer("Sigmoid"))

        self.action_maxpool = QtWidgets.QAction(self)
        self.action_maxpool.setText("Max Pool")
        self.action_maxpool.triggered.connect(lambda: self.create_layer("Max Pool"))

        self.action_avgpool = QtWidgets.QAction(self)
        self.action_avgpool.setText("Avg Pool")
        self.action_avgpool.triggered.connect(lambda: self.create_layer("Avg Pool"))

        self.action_view = QtWidgets.QAction(self)
        self.action_view.setText("View")
        self.action_view.triggered.connect(lambda: self.create_layer("View"))

        self.action_dropout = QtWidgets.QAction(self)
        self.action_dropout.setText("Dropout")
        self.action_dropout.triggered.connect(lambda: self.create_layer("Dropout"))

        self.action_picture1 = QtWidgets.QAction(self)
        self.action_picture1.setText("Picture1")
        self.action_picture1.triggered.connect(lambda: self.open_data_dialog("Picture1"))

        self.action_picture2 = QtWidgets.QAction(self)
        self.action_picture2.setText("Picture2")
        self.action_picture2.triggered.connect(lambda: self.open_data_dialog("Picture2"))

        self.action_train = QtWidgets.QAction(self)
        self.action_train.setText("Train")
        self.action_train.triggered.connect(self.open_train_dialog)

        self.action_test_set = QtWidgets.QAction(self)
        self.action_test_set.setText("Testset")
        self.action_test_set.triggered.connect(lambda: self.test_network("Testset"))

        self.action_test_pic = QtWidgets.QAction(self)
        self.action_test_pic.setText("Picture")
        self.action_test_pic.triggered.connect(lambda: self.test_network("Picture"))

        self.menu_file.addAction(self.action_load_network)
        self.menu_file.addAction(self.action_save_network)
        self.menu_file.addAction(self.action_export_py)

        self.menu_layer.addAction(self.action_conv)
        self.menu_layer.addAction(self.action_linear)
        self.menu_layer.addAction(self.action_dropout)


        self.menu_functions.addAction(self.action_relu)
        self.menu_functions.addAction(self.action_sigmoid)
        self.menu_functions.addAction(self.action_maxpool)
        self.menu_functions.addAction(self.action_avgpool)
        self.menu_functions.addAction((self.action_view))

        self.menu_data.addAction(self.action_picture1)
        self.menu_data.addAction(self.action_picture2)

        self.menu_test.addAction(self.action_test_set)
        self.menu_test.addAction(self.action_test_pic)

        self.menubar.addAction(self.menu_file.menuAction())
        self.menubar.addAction(self.menu_layer.menuAction())
        self.menubar.addAction(self.menu_functions.menuAction())
        self.menubar.addAction(self.menu_data.menuAction())
        self.menubar.addAction(self.action_train)
        self.menubar.addAction(self.menu_test.menuAction())

        """
        self.toolBar = self.addToolBar("Toolbar")
        self.toolBar.addAction(self.action_linear)
        self.toolBar.addAction(self.action_conv)
        """

        self.retranslateUi()
        #self.connect(self.data_thread, QtCore.SIGNAL("create_console_text"), self.create_console_text)
        self.data_thread.signal.connect(self.create_console_text)
        self.network_thread.signal.connect(self.create_console_text)
        self.network_thread.signal2.connect(self.save_network_by_train)
        self.world.signal.connect(self.reset_weights)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle("Networkx")
        #self.console_start_text.setText(_translate("MainWindow", "TextLabel"))

    def mouseMoveEvent(self, event):
        if (self.resizing ==  True):
            movement = event.pos() - self.mouse_pos
            self.mouse_pos = event.pos()
            self.scroll_console.setMaximumSize(self.scroll_console.size()-QtCore.QSize(0,movement.y()))

        elif (event.y() >= self.scroll_world.y() + self.scroll_world.height()+30 and
                event.y() <= self.scroll_console.y() + 30 and event.x() >= self.scroll_world.x() and
                event.x() <= self.scroll_world.x() + self.scroll_world.width()):
            self.setCursor(QtCore.Qt.SizeVerCursor)
        else:
            self.setCursor(QtCore.Qt.ArrowCursor)

    def mousePressEvent(self, event):
        if (event.y() >= self.scroll_world.y() + self.scroll_world.height() and
                event.y() <= self.scroll_console.y() + 30 and event.x() >= self.scroll_world.x() and
                event.x() <= self.scroll_world.x() + self.scroll_world.width()):
            self.mouse_pos = event.pos()
            self.resizing = True

    def mouseReleaseEvent(self, event):
        self.resizing = False

    def create_layer(self, type):
        """create layer for worldobject of type type"""

        layer = self.world.network_layers
        connections = self.world.connections
        first_layer = self.world.first_layer
        self.world = World(layer, connections,first_layer)
        self.world.add_layer(type)
        self.scroll_world.setWidget(self.world)

    def create_text(self):
        """only for test xD"""
        for i in range(500):
            self.create_console_text(" ")

    def create_console_text(self,text):
        """create text in console"""

        self.console.add_text(text)
        self.console.setMinimumSize(self.console.size() + QtCore.QSize(0,20))
        #self.scroll_console.setWidget(self.console)

    def clear_console(self):
        """delete all text of console"""

        self.console = Console([])
        self.scroll_console.setWidget(self.console)

    def open_data_dialog(self,type):
        """open dialog for create training/test data"""

        #file_dialog = QtWidgets.QFileDialog()
        #fname = file_dialog.getOpenFileName(self, "Select File", "C:\\UsersPasca\\Documents")# "nur bilddateien (*.jpg *.png)")
        self.data_thread.type = type
        self.data_thread.path_train = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Training Directory"))
        self.data_thread.path_test = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Test Directory"))
        if(type == "Picture2"):
            self.dialog = Data_Dialog(True,True)
        else:
            self.dialog = Data_Dialog(True,False)
        self.dialog.buttonBox.accepted.connect(self.load_data)

    def load_data(self):
        """load settings from data dialog to data_thread and start creating train/test data"""

        self.data_thread.batch_size = int(self.dialog.batch_spinBox.value())
        self.data_thread.resize = int(self.dialog.resize_spinBox.value())
        if(self.dialog.norm_mean_input.text() == "" and self.dialog.norm_std_input.text() == ""):
            #print("empty")
            self.data_thread.transforms = transforms.Compose([transforms.Resize(self.data_thread.resize),  # 256*256
                                                              transforms.CenterCrop(self.data_thread.resize),
                                                              # schneidet im zentrum ab
                                                              transforms.ToTensor(), ])
        else:
            self.data_thread.normalize = transforms.Normalize(
                mean=self.make_mean_std(self.dialog.norm_mean_input.text()),
                std=self.make_mean_std(self.dialog.norm_std_input.text())
            )

            self.data_thread.transforms = transforms.Compose([transforms.Resize(self.data_thread.resize),  # 256*256
                                                              transforms.CenterCrop(self.data_thread.resize),
                                                              # schneidet im zentrum ab
                                                              transforms.ToTensor(),
                                                              self.data_thread.normalize])

        if(self.data_thread.type == "Picture2"):
            self.data_thread.categories = self.dialog.cat_input.text().split(",")
            #print(self.data_thread.categories)
        self.dialog.reject()
        #self.create_data(self.data_controller.train_data, self.data_controller.path_train, self.data_controller.batch_size, self.data_controller.transforms)
        self.data_thread.start()
        #self.loop()

    def make_mean_std(self,string):
        """creates mean/std list for nomalize pictures"""

        help_list = string.split(",")
        for i in range(len(help_list)):
            #print(help_list[i])
            help_list[i] = float(help_list[i].strip(" "))
            #print(help_list[i])
        #print(help_list)
        return help_list

    def open_train_dialog(self):
        """open dialog for train neural network"""

        if(self.data_thread.train_data != []):
            self.dialog = Train_Dialog()
            self.dialog.buttonBox.accepted.connect(self.train_network)
        else:
            self.create_console_text("You have to load data to train!")

    def load_network(self):
        """load network from .nx file"""

        self.world.connections = []
        self.world.network_layers = []
        first = True
        fd = QtWidgets.QFileDialog.getOpenFileName(self,"Load Network", "", "network (*.nx)")
        layer_load = []
        #print("looooooooad network")
        if (fd != ('', '')):
            file = open(fd[0], "r")
            for line in file:
                if(first):
                    connections = eval(line)[0]
                    #print(connections)
                    first_layer = eval(line)[1]
                    #print(first_layer,type(first_layer))
                    path_weights = ""
                    if(eval(line)[2] != "path of current weights"):
                        for path in fd[0].split("/")[:-1]:
                            path_weights += path + "/"
                            #print(path_weights)
                        path_weights += eval(line)[2]
                    else:
                        path_weights = "path of current weights"
                    #print(path_weights)
                    self.network_thread.path_weights = path_weights
                    self.network_thread.categories = eval(line)[3]
                    first = False
                else:
                    layer_load.append(eval(line))
        #print(layer_load)
        for elem in layer_load:
            self.create_layer(elem[0][0])
        self.world.set_layer_settings(connections, layer_load, first_layer)

    def save_network(self):
        """save network to .nx file"""

        save_list = self.world.create_list_for_save()
        connections = self.world.connections
        first_layer = self.world.first_layer
        #print("looooo")
        fd = QtWidgets.QFileDialog.getSaveFileName(self,"Save Network", "" , "network (*.nx)")
        if(fd != ('','')):
            #print(fd)
            file = open(fd[0], "w")
            #print(1)
            file.write(str((connections,first_layer,self.network_thread.path_weights,self.network_thread.categories)) +"\n")
            #print(2)
            for layer in save_list:
                file.write(str(layer) + "\n")
            file.close()

    def save_network_by_train(self,path):
        """save network to .nx file to path path """
        save_list = self.world.create_list_for_save()
        connections = self.world.connections
        first_layer = self.world.first_layer
        #print("looooo")
        fd = path[:-3] + "_Network.nx"
        #print(path)
        #print(fd)
        if(fd != ('','')):
            #print(fd)
            file = open(fd, "w")
            #print(1)
            file.write(str((connections,first_layer,path.split("/")[-1],self.network_thread.categories)) +"\n")
            #print(2)
            for layer in save_list:
                file.write(str(layer) + "\n")
            file.close()
            #print(3)

    def reset_weights(self,nonsense):
        """randomize weights of neural network"""
        #print("reset weights")
        self.network_thread.path_weights = "path of current weights"

    def train_network(self):
        """start training of neural network"""
        #print("start train network function")
        #print(self.network_thread.path_weights)
        self.load_model()
        #print("model check")
        #print(self.dialog.optimizer_comboBox.currentText())
        self.network_thread.set_optimizer(self.dialog.optimizer_comboBox.currentText(),self.dialog.lr_line_edit.text(),self.dialog.momentum_line_edit.text())
        #self.network_thread.set_optimizer("SGD")
        #print(self.network_thread.optimizer)
        self.network_thread.set_criterion(self.dialog.loss_comboBox.currentText())
        #print(self.network_thread.criterion)
        self.network_thread.epochs = self.dialog.epochs_spinBox.value()
        #print(self.network_thread.epochs)
        self.network_thread.use_gpu = self.dialog.gpu_radio_Button.isChecked()
        #print(self.network_thread.use_gpu)
        self.network_thread.train_data = self.data_thread.train_data
        self.network_thread.test_data = self.data_thread.test_data
        self.network_thread.batch_size = self.data_thread.batch_size

        #self.network_thread.categories = listdir(self.data_thread.path_train)
        self.network_thread.categories = self.data_thread.categories

        #print("everything okay")
        self.dialog.reject()
        self.network_thread.save_path = QtWidgets.QFileDialog.getExistingDirectory()
        
        self.network_thread.start()

    def test_network(self,type):
        """test neural network in type way"""

        self.load_model()
        if(type == "Testset"):
            if(self.data_thread.test_data != []):
                self.network_thread.test_data = self.data_thread.test_data
                self.network_thread.batch_size = 64 ########################################################################
                self.network_thread.test()
            else:
                self.create_console_text("You have to load data to test!")

        elif(type == "Picture"):
            self.dialog = Data_Dialog(False,False)
            self.dialog.buttonBox.accepted.connect(self.Test_Picture)

    def Test_Picture(self):
        """predict picture with neural network and open dialog for testing picture"""

        if (self.dialog.norm_mean_input.text() == "" and self.dialog.norm_std_input.text() == ""):
            #print("empty")
            transform = transforms.Compose([transforms.Resize(self.dialog.resize_spinBox.value()),  # 256*256
                                                              transforms.CenterCrop(self.dialog.resize_spinBox.value()),
                                                              # schneidet im zentrum ab
                                                              transforms.ToTensor(), ])
        else:
            normalize = transforms.Normalize(
                mean=self.make_mean_std(self.dialog.norm_mean_input.text()),
                std=self.make_mean_std(self.dialog.norm_std_input.text())
            )

            transform = transforms.Compose([transforms.Resize(self.dialog.resize_spinBox.value()),  # 256*256
                                                              transforms.CenterCrop(self.dialog.resize_spinBox.value()),
                                                              # schneidet im zentrum ab
                                                              transforms.ToTensor(),
                                                              normalize])
        fname = QtWidgets.QFileDialog.getOpenFileName(self, "Select File", "C:", "nur bilddateien (*.jpg *.png)")
        #print(fname[0])
        img = Image.open(fname[0])
        img_tensor = transform(img)
        #print(img_tensor.size())
        #print(img_tensor)
        data = []
        data.append(img_tensor)
        #print("stack")
        data = torch.stack(data)
        #print(data.size())
        if(next(self.network_thread.model.parameters()).is_cuda):
            data = data.cuda()
        data = Variable(data)

        #print(data)
        out =  self.network_thread.model(data)
        prediction = out.data.max(1, keepdim=True)[1].item()
        ###########################################################################################################
        self.dialog = Test_Picture_Dialog(fname[0], self.network_thread.categories[prediction])

    def load_model(self):
        """load neural network with random or saved weights"""

        if (self.network_thread.path_weights == "path of current weights"):#random weights
            self.network_thread.model = Network(self.world.get_network())
        else:#saved weights
            print("UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUSEEEEEEEEEEEEEEEEE WEIGHTS")
            if (os.path.isfile(self.network_thread.path_weights)):
                self.network_thread.model = torch.load(self.network_thread.path_weights)

    def export_network_as_py(self):
        """export .py file from neural network for using network outside"""
        ############################################################
        fd = QtWidgets.QFileDialog.getSaveFileName(self, "Export Network", "", "Python (*.py)")
        #print("1")
        path_weights = fd[0][:-3] + "_weights.pt"
        #print("2")
        resize = self.data_thread.resize
        #print("3")
        py_data =["import torch", "from torchvision import transforms", "from PIL import Image",
         "from torch.autograd import Variable", "import torch.nn.functional as F",
         "import torch.nn as nn"," ", "path_weights = '" + str(path_weights) +"'",
         "path_picture =  'Your Path here'", "resize = " + str(resize)," ",
         "categories = " + str(self.network_thread.categories),
         "transform = transforms.Compose([transforms.Resize(resize),",
         "                                transforms.CenterCrop(resize),",
         "                                transforms.ToTensor(), ])",
         " ", "class Network(nn.Module): ", "    def __init__(self,layer_list):",
         "        super(Network,self).__init__()", "        self.functions = []",
         "        layers = []", "        for layer in layer_list:",
         "            self.functions.append(self.create_layer(layer))",
         "        for layer in self.functions:", "            if(not isinstance(layer, list)):",
         "                layers.append(layer)", "        self.layer = nn.ModuleList(layers)",
         " ", "    def forward(self, x):", "        layer_num = 0", "        for layer in self.functions:",
         "            if(isinstance(layer, list)):", "                if(layer[0] == 'ReLu'):",
         "                    x = F.relu(x)", "                elif(layer[0] == 'Sigmoid'):",
         "                    x = torch.sigmoid(x)", "                elif(layer[0] == 'Max Pool'):",
         "                    x = F.max_pool2d(x, layer[1])", "                elif(layer[0] == 'Avg Pool'):",
         "                    x = F.avg_pool2d(x, layer[1])", "                elif(layer[0] == 'View'):",
         "                    x = x.view(-1,layer[1])", "            else:",
         "                x = self.layer[layer_num](x)", "                layer_num += 1", "        return x",
         " ", "model = torch.load(path_weights)", " ", "img = Image.open(path_picture)",
         "img_tensor = transform(img)", "data = []", "data.append(img_tensor)",
         "data = torch.stack(data) #create Tensor([1,1,resize,resize])","if(next(model.parameters()).is_cuda):",
         "    data = data.cuda()", "data = Variable(data)",
         "out =  model(data) #network output", "prediction = out.data.max(1, keepdim=True)[1].item()",
         "print(categories[prediction])"]
        #print(fd)
        #print(path_weights)
        if (fd != ('', '')):
            torch.save(self.network_thread.model, path_weights)
            #print("model saved")
            file = open(fd[0], "w")
            for line in py_data:
                file.write(line + "\n")
            file.close()


if __name__ == "__main__":
    
    #print(1)
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = Main_Window()
    #print(2)
    MainWindow.showMaximized()
    #print(3)
    sys.exit(app.exec_())
