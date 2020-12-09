import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from efficientnet_pytorch import EfficientNet

sigmoid = nn.Sigmoid()

# Swish activation function
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

# Model Definition
class EfficientNet_Models(nn.Module):
    def __init__(self, output_size, num_cols, model_name):
        super(EfficientNet_Models, self).__init__()
        self.output_size = output_size
        self.num_cols = num_cols
        self.features = EfficientNet.from_pretrained(model_name)
        self.features._fc = nn.Linear(in_features=1408, out_features=500, bias=True)
        
        # nn.Linear(4) is number of columns in csv 
        self.fcn_model= nn.Sequential(nn.Linear(self.num_cols, 500),
                                 nn.BatchNorm1d(500),
                                 Swish_Module(),
                                 nn.Dropout(p=0.3), 

                                 nn.Linear(500, 250),
                                 nn.BatchNorm1d(250),
                                 Swish_Module(),
                                 nn.Dropout(p=0.3))

        self.final_model = nn.Linear(500+250,output_size)

    def forward(self, img_data, csv_data):
        img_output = self.features(img_data)
        csv_output = self.fcn_model(csv_data)

        final_data= torch.cat((img_output, csv_output), dim=1)

        final_output= self.final_model(final_data)
        return final_output


class Resnet_Model(nn.Module):
    def __init__(self, output_size, num_cols):
        super(Resnet_Model, self).__init__()
        self.output_size = output_size
        self.num_cols = num_cols
        self.features = resnet50(pretrained=True) # 1000 neurons out
        
        # nn.Linear(4) is number of columns in csv 
        self.fcn_model= nn.Sequential(nn.Linear(self.num_cols, 500),
                                 nn.BatchNorm1d(500),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.3),

                                 nn.Linear(500, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.3),

                                 nn.Linear(250, 125),
                                 nn.BatchNorm1d(125),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.3))

        self.final_model = nn.Sequential(nn.Linear(1000+125, output_size))

    def forward(self, img_data, csv_data):
        img_output = self.features(img_data)
        csv_output = self.fcn_model(csv_data)

        final_data= torch.cat((img_output, csv_output), dim=1) # 1000 + 100

        final_output= self.final_model(final_data)
        return final_output