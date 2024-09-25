import torch
import numpy as np
import collections
import torchvision
import random
import copy
import time
from torchvision.transforms.functional import rotate
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim 
import torch.nn.functional 
import torchvision.datasets   
import torchvision.transforms   

transformImg = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

def check_cov_list(train_cov_dict):
    if len(train_cov_dict) == 4694:
        return "Congratulation! You have passed the sanity check"
    else:
        return "It seems there are some errors."
    
def turn_numpy_into_torch_array(input_list):
    result = list()
    for item in input_list:
        if isinstance(item, np.ndarray):
            result.append(torch.from_numpy(item).float())
        else:
            result.append(item)
    return result

def get_training_loader(idx_path, data_path='./data', batch_size=64):
    train = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transformImg)
    train_idx = torch.load(idx_path)
    train_set = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, sampler=train_set)
    return train_loader

def get_test_loader(data_path='./data', batch_size=64):
    test = torchvision.datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transformImg)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)
    return test_loader
    
def load_model(arch, path):
    model = arch()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)
    model.float()
    return model

def image_to_tensor(image):
    return torch.tensor(image).unsqueeze(0).unsqueeze(0).float()

'''
def image_translation(img, params):
    rows, cols, ch = img.shape
    M = np.float32([[1, 0, params], [0, 1, params]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst
'''

def image_scale(img, params):
    rows, cols, ch = img.shape
    res = cv2.resize(img, None, fx=params, fy=params, interpolation=cv2.INTER_CUBIC)
    res = res.reshape((res.shape[0],res.shape[1],ch))
    y, x, z = res.shape
    if params > 1:  # need to crop
        startx = x // 2 - cols // 2
        starty = y // 2 - rows // 2
        return res[starty:starty + rows, startx:startx + cols]
    elif params < 1:  # need to pad
        sty = int((rows - y) / 2)
        stx = int((cols - x) / 2)
#             print((sty, rows - y - sty), (stx, cols - x - stx),np.array([(sty, rows - y - sty), (stx, cols - x - stx), (0, 0)]).dtype.kind)
        return np.pad(res, [(sty, rows - y - sty), (stx, cols - x - stx), (0, 0)], mode='constant',
                        constant_values=0)
    return res

def image_shear(img, params):
    rows, cols, ch = img.shape
    factor = params * (-1.0)
    M = np.float32([[1, factor, 0], [0, 1, 0]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def image_rotation(img, params):
    new_img  = rotate(torch.tensor(img), params)
    return new_img.numpy()

"""
def image_rotation(img, params):
    rows, cols, ch = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), params, 1)
    dst = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_AREA)
    return dst
"""

def image_contrast(img, params):
    alpha = params
    new_img = cv2.multiply(img, np.array([alpha]))  # mul_img = img*alpha

    return new_img

def image_brightness(img, params):
    # Attention: Only works for 1 channel image.
    beta = params
    sec_matrix = np.zeros(img.shape)
    sec_matrix += beta
    if beta >= 0:
        mask = (img >= beta)
    else:
        mask = (img <= 255 + beta)
    new_img = img - beta * mask # new_img = img*alpha + beta
    return new_img

def image_blur(img, params):
    img = img.transpose([1, 2, 0])
    img = img.astype('uint8')
    # print("blur")
    blur = []
    if params == 1:
        blur = cv2.blur(img, (3, 3))
    if params == 2:
        blur = cv2.blur(img, (4, 4))
    if params == 3:
        blur = cv2.blur(img, (5, 5))
    if params == 4:
        blur = cv2.GaussianBlur(img, (3, 3), 0)
    if params == 5:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
    if params == 6:
        blur = cv2.GaussianBlur(img, (7, 7), 0)
    if params == 7:
        blur = cv2.medianBlur(img, 3)
    if params == 8:
        blur = cv2.medianBlur(img, 5)
    # if params == 9:
    #     blur = cv2.blur(img, (6, 6))
    if params == 9:
        blur = cv2.bilateralFilter(img, 6, 50, 50)
        # blur = cv2.bilateralFilter(img, 9, 75, 75)
    blur = np.expand_dims(blur, 0)
    #blur = blur.transpose([2, 0, 1])
    return blur

def image_pixel_change(img, params):
    # random change 1 - 5 pixels from 0 -255
    img_shape = img.shape
    img1d = np.ravel(img)
    arr = np.random.randint(0, len(img1d), params)
    for i in arr:
        img1d[i] = np.random.randint(0, 256)
    new_img = img1d.reshape(img_shape)
    return new_img

def image_noise(img, params):
    if params == 1:  # Gaussian-distributed additive noise.
        row, col, ch = img.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5

        gauss = np.random.normal(mean, sigma, (row, col, ch))*255
        mask = np.random.uniform(0,1,(row, col, ch))
        noisy = np.clip(img + mask*gauss,0,255)
        return noisy.astype(np.uint8)
    elif params == 2:  # Replaces random pixels with 0 or 1.
        s_vs_p = 0.5
        amount = 0.005
        out = np.copy(img)
        # Salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i, int(num_salt))
                    for i in img.shape]
        out[tuple(coords)] = 255
        # Pepper mode
        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i, int(num_pepper))
                    for i in img.shape]
        out[tuple(coords)] = 0
        return out
    elif params == 3:  # Multiplicative noise using out = image + n*image,where n is uniform noise with specified mean & variance.
        row, col, ch = img.shape
        gauss = np.random.randn(row, col, ch)
        mask = np.random.uniform(0,1,(row, col, ch))
        gauss *= mask
        noisy = np.clip(img + img * gauss,0,255)
        return noisy.astype(np.uint8)
        
class Mutators():
    def __init__(self):
        self.transformations = [
            image_scale, image_shear, image_rotation,
            image_contrast, image_brightness, image_blur, image_pixel_change, image_noise
            ]
        self.classA = [6, 7]  # pixel value transformation
        self.classB = [0, 1, 2, 3, 4, 5] # Affine transformation
        params = []
        #params.append(list(range(-3, 3)))  # image_translation
        params.append(list(map(lambda x: x * 0.1, list(range(7, 12)))))  # image_scale
        params.append(list(map(lambda x: x * 0.1, list(range(-6, 6)))))  # image_shear
        params.append(list(range(-50, 50)))  # image_rotation
        params.append(list(map(lambda x: x * 0.1, list(range(5, 13)))))  # image_contrast
        params.append(list(range(-20, 20)))  # image_brightness
        params.append(list(range(1, 10)))  # image_blur
        params.append(list(range(1, 10)))  # image_pixel_change
        params.append(list(range(1, 4)))  # image_noise
        self.params = params  

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