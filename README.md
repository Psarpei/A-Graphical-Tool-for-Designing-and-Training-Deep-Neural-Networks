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
    <img align="top" width="400" height="" src="https://upload.wikimedia.org/wikipedia/commons/3/3d/NeuronalesNetzErstellen1.jpg">
</p>

The building block can be moved like any other by holding down the mouse button and dragging it to the area of the building block. To change the parameters, the mouse must also be placed on the area of the building block and the right mouse button must be pressed.  A drop-down menu appears, which allows to execute the functions Edit, Draw, Delete, Delete links and First Layer. When Draw is pressed, the mouse can be used, left-click on another block to create a connection with it. By double-clicking with the left mouse button on a block, a connection can also be made from this block. The connection is made from the output of the block from which a connection is made to the input of the block to which the connection is to which the connection is subsequently closed with a left click. By pressing the left mouse button on the same block the creation of a connection can be canceled. Each block can have only one connection at the output and one at the input. An exception is the First Layer, which cannot have a connection at the input. The Delete function is used to delete the block. Delete Links deletes the connections of the block both at the input and at the output. By selecting First Layer, the current POU becomes the First Layer, turns green and is interpreted by the software as the input layer of the network when generating the artificial neural network. To edit the parameters of a modifiable block, press Edit.

<p align="center">                                                                                                                    
    <img align="top" width="400" height="" src="https://upload.wikimedia.org/wikipedia/commons/c/c0/NeuronalesNetzErstellen2.jpg">
</p>

To change the values, a popup window opens where the values can be selected. With the Ok button the values are taken over in the module, with the Cancel button the change is discarded. the change is discarded. In this case, the parameter in is set to 1, because the MNIST data are gray scale images, which have only one channel.
In order to add a Max-Pooling block the entry Max-Pool, as shown in Fig.4.8. The parameter kernel is changed in the same way as for the Convolution Layer. In this case a 2x2 kernel is needed for the desired architecture. Afterwards the Convolutional Layer and the Max-Pooling have to be connection from the output of the convolutional layer to the input. This is done as described before and should be done after moving the max-pooling block under the convolutional layer block. Now it should look like in the figure above.

**The neural Network**

The final convolutional neural network used here is shown below. 

<p align="center">                                                                                                                    
    <img align="top" width="1200" height="" src="https://upload.wikimedia.org/wikipedia/commons/3/34/NeuronalesNetzErstellen6.jpg">
</p>

The network can be created exactly as described before.

**Read in Data**

In order to supply the network with data for training, the data must be read in and preprocessed.  In the current state, the Data submenu contains 2 different types for processing image data. 

<p align="center">                                                                                                                    
    <img align="top" width="300" height="" src="https://upload.wikimedia.org/wikipedia/commons/b/b1/NeuronalesNetzErstellen7.jpg">
</p>

The basic difference between Picture1 and Picture2 is the way the data is stored. When one of the two is selected, a file dialog opens, which prompts to select the folder with the training data.

<p align="center">                                                                                                                    
    <img align="top" width="1000" height="" src="https://upload.wikimedia.org/wikipedia/commons/f/fc/NeuronalesNetzErstellen8.jpg">
</p>

Im Fall von Picture1 müssen die Bilder für das Training wie folgt gespeichert sein.

<p align="center">                                                                                                                    
    <img align="top" width="1000" height="" src="https://upload.wikimedia.org/wikipedia/commons/4/48/NeuronalesNetzErstellen9.jpg">
</p>

Here the images are sorted into their own subfolders based on their labels. In this case, all zeros in the Zero folder, all ones in the One folder, and so on. Later, the neural network uses exactly the names of the individual folders as labels for the prediction. With Picture2 the whole thing looks a bit different. Here are all training data strored in one folder.

<p align="center">                                                                                                                    
    <img align="top" width="1000" height="" src="https://upload.wikimedia.org/wikipedia/commons/f/fc/NeuronalesNetzErstellen8.jpg">
</p>

The labels of the individual images are contained in the name of the image, which is irrelevant for Picture1. After selecting the training and test data, another popup window opens in which the individual categories for the labels can be entered. The software then automatically assigns the appropriate class to each
class as label, if the name of a class is contained in the title, as shown in the following on the basis of the Cats and Dog dataset.

<p align="center">                                                                                                                    
    <img align="top" width="1000" height="" src="https://upload.wikimedia.org/wikipedia/commons/7/70/NeuronalesNetzErstellen10.jpg">
</p>

After selecting the training dataset, a file dialog opens again, which now requests the test dataset. The same format applies for this as for the training data set.
Last but not least the already mentioned popup window will open.

<p align="center">                                                                                                                    
    <img align="top" width="300" height="" src="https://upload.wikimedia.org/wikipedia/commons/e/e1/NeuronalesNetzErstellen11.jpg">
</p>

