import torch
import torch.nn as nn
import torch.optim 
import torch.nn.functional 
import torchvision.datasets   
import torchvision.transforms     

# Defining the network (LeNet-5)  
# Code reference: https://github.com/bollakarthikeya/LeNet-5-PyTorch/blob/master/lenet5_gpu.py

class Lenet5(nn.Module):

    def __init__(self, num_classes=10, dropRate=0.0, exclude_layer = ['linear','conv']):
        super(Lenet5, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        
        
        self.fc_1 = nn.Linear(4*4*16, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.relu = nn.ReLU()
        self.fc_3 = nn.Linear(84, num_classes)
        self.dropout = dropRate
        self.exclude_layer = exclude_layer
        
     
    def forward(self, x):
        """
        :param x: [b, 1, 28, 28]
        :return:
        """
        batchsz = x.size(0)
        x = self.conv_1(x)
        if self.dropout > 0:
            x = F.dropout(x,p = self.dropout)
        x = self.conv_2(x)
        if self.dropout > 0:
            x = F.dropout(x,p = self.dropout)
        
        x = x.view(batchsz, 4*4*16)
        
        x = self.fc_1(x)
        if self.dropout > 0:
            x = F.dropout(x,p = self.dropout)
        x = self.fc_2(self.relu(x))
        if self.dropout > 0:
            x = F.dropout(x,p = self.dropout)
        logits = self.fc_3(self.relu(x))
        return logits
    
    
    # function to extract the multiple features(before activations)
    def feature_list(self, x):
        batchsz = x.size(0)
        out_list = []
        name_list = ["conv1","conv2","fc1","fc2","logits"]
        
        for name,module in self.conv_1.named_children():
            x = module(x)
            if any(item in str(type(module)).lower() for item in self.exclude_layer):
                out_list.append(x)

        for name,module in self.conv_2.named_children():
            x = module(x)
            if any(item in str(type(module)).lower() for item in self.exclude_layer):
                out_list.append(x)
        
        x = x.view(batchsz, 4*4*16)
        x = self.fc_1(x)
        out_list.append(x)
        x = self.relu(x)
        x = self.fc_2(x)
        out_list.append(x)
        x = self.relu(x)
        logits = self.fc_3(x)
        out_list.append(logits)
        
        return logits, out_list, name_list
    
    
    def intermediate_forward(self, x, layer_index):
        batchsz = x.size(0)
        if layer_index == 0:
            out = self.conv_1(x)
        elif layer_index == 1:
            out = self.conv_1(x)
            out = self.conv_2(out)
        elif layer_index == 2:
            out = self.conv_1(x)
            out = self.conv_2(out)
            out = out.view(batchsz, 4*4*16)
            out = self.relu(self.fc_1(out))
        
        elif layer_index == 3:
            out = self.conv_1(x)
            out = self.conv_2(out)
            out = out.view(batchsz, 4*4*16)
            out = self.relu(self.fc_1(out))
            out = self.relu(self.fc_2(out))
            
        elif layer_index == 4:
            out = self.conv_1(x)
            out = self.conv_2(out)
            out = out.view(batchsz, 4*4*16)
            out = self.relu(self.fc_1(out))
            out = self.relu(self.fc_2(out))
            out = self.fc_3(out)
        
        else:
            assert False,"wrong layer index"
        return out