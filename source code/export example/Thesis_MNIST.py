import torch
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
 
path_weights = 'C:/Users/Pasca/Dropbox/Bachelorarbeit/Thesis_MNIST_weights.pt'
path_picture =  'Your Path here'
resize = 28
 
categories = ['Eight', 'Five', 'Four', 'Nine', 'One', 'Seven', 'Six', 'Three', 'Two', 'Zero']
transform = transforms.Compose([transforms.Resize(resize),
                                transforms.CenterCrop(resize),
                                transforms.ToTensor(), ])
 
class Network(nn.Module): 
    def __init__(self,layer_list):
        super(Network,self).__init__()
        self.functions = []
        layers = []
        for layer in layer_list:
            self.functions.append(self.create_layer(layer))
        for layer in self.functions:
            if(not isinstance(layer, list)):
                layers.append(layer)
        self.layer = nn.ModuleList(layers)
 
    def forward(self, x):
        layer_num = 0
        for layer in self.functions:
            if(isinstance(layer, list)):
                if(layer[0] == 'ReLu'):
                    x = F.relu(x)
                elif(layer[0] == 'Sigmoid'):
                    x = torch.sigmoid(x)
                elif(layer[0] == 'Max Pool'):
                    x = F.max_pool2d(x, layer[1])
                elif(layer[0] == 'Avg Pool'):
                    x = F.avg_pool2d(x, layer[1])
                elif(layer[0] == 'View'):
                    x = x.view(-1,layer[1])
            else:
                x = self.layer[layer_num](x)
                layer_num += 1
        return x
 
model = torch.load(path_weights)
 
img = Image.open(path_picture)
img_tensor = transform(img)
data = []
data.append(img_tensor)
data = torch.stack(data) #create Tensor([1,1,resize,resize])
if(next(model.parameters()).is_cuda):
    data = data.cuda()
data = Variable(data)
out =  model(data) #network output
prediction = out.data.max(1, keepdim=True)[1].item()
print(categories[prediction])
