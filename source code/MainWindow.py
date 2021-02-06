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
