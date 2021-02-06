# Networkx - a Graphical Tool for Designing and Training Deep Neural Networks
This work is the result of my bachelor thesis. The idea was to build a graphical tool for design, create, and test neural networks. Further the tool should be able to export the neural network to integrate it into other applications. On top of that it should be able to load and preprocessing the data.


## General information
<img align="right" width="300" height="" src="https://upload.wikimedia.org/wikipedia/commons/1/1e/Logo-Goethe-University-Frankfurt-am-Main.svg">

**Instructors**
* [Prof. Dr. Detlef Kroemker](https://www.studiumdigitale.uni-frankfurt.de/75926978/Prof__Dr__Detlef_Kr%C3%B6mker__Pensioniert_seit_01_04_2020)

**Institutions**
* **[Goethe University](http://www.informatik.uni-frankfurt.de/index.php/en/)**
* **[GDV - Uni Frankfurt](https://www.gdv.informatik.uni-frankfurt.de/)**

**Project team**
* [Pascal Fischer](https://github.com/Psarpei/)

**Tools**
* Python 3
* PyQT5
* PyTorch
* Pillow

## Networkx
The software currently offers the following building blocks and functions to create artificial neural networks:

* Convolutional, dropout layers as well as fully connected layers.
* Max and average pooling
* The activation functions ReLU and sigmoid
* A function to transfer the neurons from several feature maps into a feature map for further processing
* Drawing connections between the layers with automatic creation of the connections of the correct neurons between the layers, depending on the selected type of layer
* Automatic application of any number of successively executed activation functions to the output neurons of a layer, depending on the selection of the activation functions and the type of layer.
* Unlimited area for adding neural network building blocks in the worldwidget.
* Automatic creation of an artificial neural network from the connected layers within the worldwidget

In addition to the building blocks and functions for creating artificial neural networks, the software contains the functions:

* Loading and saving artificial neural networks and their current weights.
* Training of artificial neural networks
* 2 different types for reading and automatic preprocessing of image data and their labels for training a neural network for object classification in images.
* Training artificial neural networks for classification of images.
* Test and evaluation of artificial neural networks for the classification of images with single images.
* Test and evaluation of datasets during training as well as outside of training.
* Exporting trained artificial neural networks to an executable Python script.
* Output of instructions on how to operate the software, the current training status, test results etc. in the console.

**GUI**

<p align="center">                                                                                                                    
    <img align="top" width="1300" height="" src="https://upload.wikimedia.org/wikipedia/commons/3/3e/Networkx1.jpg">
</p>

The basic part of the user interface consists of the so-called world widget, which represents the white horizontally and vertically scrollable area, and the console, which represents the white area directly below it. In addition, there is a toolbar directly at the top of the software, which provides functions for using the software in certain categories. The Worldwidget, through the submenus Layer and Functions, building blocks for the generation of artificial neural networks can be added. In the Worldwidget the
blocks can be linked to a graph and modified by their possible settings. settings. The input layer is marked in green and can be changed at any time. During the training or testing process, the software automatically creates an artificial neural network from the graph and processes the weights invisibly for the user. The console has the task of providing the user information about incorrect operation, training status, test results, etc. If the console size is not capable of displaying all previous outputs, it becomes scrollable, just like the Worldwidget. With the clear button on the left below the console can be used to clear the contents of the console. Besides the submenus
Layer and Functions, the software contains 4 more submenus. The File submenu, which provides functions for importing and exporting neural networks. The data submenu, which allows loading and processing of training data. The submenu Train which provides functions for training (currently only one type of training with different
training with different settings) and the submenu Test, which provides functions to provides functions for evaluation and testing. The size of the Worldwidgets and the
console can be resized by using the mouse on the gray border between them.

## Tutorial

In the following the use of Networkx will be explained by creating a convolutional neural network for recognizing digits with the [MNIST-Dataset](http://yann.lecun.com/exdb/mnist/).

**Creating a neural network**

The first step is to create a convolution layer. To add a convolution layer to the worldwidget, select the Layer submenu and click on the Convolution Layer tab. Now the convolution layer should appear in the upper left corner of the worldwidget. 


<p align="center">                                                                                                                    
    <img align="top" width="500" height="" src="https://upload.wikimedia.org/wikipedia/commons/3/3d/NeuronalesNetzErstellen1.jpg">
</p>

The building block can be moved like any other by holding down the mouse button and dragging it to the area of the building block. To change the parameters, the mouse must also be placed on the area of the building block and the right mouse button must be pressed.  A drop-down menu appears, which allows to execute the functions Edit, Draw, Delete, Delete links and First Layer. When Draw is pressed, the mouse can be used,
left-click on another block to create a connection with it.


