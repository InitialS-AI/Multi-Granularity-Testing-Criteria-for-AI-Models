import torch
import numpy as np
import collections
import torchvision
import random
import copy
import time
from mutator import Mutators

transformImg = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

def get_training_loader(data_path='./data', batch_size=64):
    train = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transformImg)
    train_idx = torch.load("train_idx.pt")
    train_set = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, sampler=train_set)
    return train_loader

def get_test_loader(data_path='./data', batch_size=64):
    test = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transformImg)
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

def mutate_one(ref_img, 
               img, 
               has_Affine, 
               l0_ref, 
               linf_ref,
               alpha=0.02,
               beta=0.20, 
               try_num=50):
    """
    Mutate the image for once.
    Line 5 ~ 18 in Algorithm 2 in the original paper.
    Args
    ---
        ref_img: reference image. s0' in the original paper.
        img: the seed. s in the original paper.
        
    """
    # ref_img is the reference image, img is the seed

    # cl means the current state of transformation
    # 0 means it can select both of Affine and Pixel transformations
    # 1 means it only select pixel transformation because an Affine transformation has been used before

    # l0_ref, linf_ref: if the current seed is mutated from affine transformation, we will record the l0, l_inf
    # between initial image and the reference image. i.e., L0(s_0,s_{j-1}) L_inf(s_0,s_{j-1}) in Equation 2 of the paper

    # tyr_num is the maximum number of trials in Algorithm 2


    x, y, z = img.shape

    # a, b is the alpha and beta in Equation 1 in the paper

    # l0_threshold: alpha * size(s), l_infinity_threshold: beta * 255 in Equation 1
    l0_threshold = int(alpha * x * y * z)
    l_infinity_threshold = int(beta * 255)

    ori_shape = ref_img.shape
    for ii in range(try_num):
        random.seed(time.time())
        if has_Affine == 0:  # 0: can choose class A and B
            tid = random.sample(Mutators.classA + Mutators.classB, 1)[0]
            # Randomly select one transformation   Line-7 in Algorithm2
            transformation = Mutators.transformations[tid]
            params = Mutators.params[tid]
            # Randomly select one parameter Line 10 in Algo2
            param = random.sample(params, 1)[0]

            # Perform the transformation  Line 11 in Algo2
            img_new = transformation(copy.deepcopy(img), param)
            img_new = img_new.reshape(ori_shape)
            
            
            # check whether it is a valid mutation. i.e., Equation 1 and Line 12 in Algo2
            sub = ref_img - img_new
            l0_ref = np.sum(sub != 0)
            linf_ref = np.max(abs(sub))
            if l0_ref < l0_threshold or linf_ref < l_infinity_threshold:
                if tid in Mutators.classA:
                    return ref_img, img_new, False, True, l0_ref, linf_ref
                else:  # B, C
                    # If the current transformation is an Affine trans, we will update the reference image and
                    # the transformation state of the seed.
                    ref_img = transformation(copy.deepcopy(ref_img), param)
                    ref_img = ref_img.reshape(ori_shape)
                    return ref_img, img_new, True, True, l0_ref, linf_ref
            
        elif has_Affine == 1: # 0: can choose class A
            tid = random.sample(Mutators.classA, 1)[0]
            transformation = Mutators.transformations[tid]
            params = Mutators.params[tid]
            param = random.sample(params, 1)[0]
            img_new = transformation(copy.deepcopy(img), param)
            sub = ref_img - img_new

            # To compute the value in Equation 2 in the paper.
            l0_new = l0_ref +  np.sum(sub != 0)
            linf_new = max(linf_ref, np.max(abs(sub)))

            if  l0_new < l0_threshold or linf_new < l_infinity_threshold:
                return ref_img, img_new, 1, 1, l0_ref, linf_ref
    # Otherwise the mutation is failed. Line 20 in Algo 2
    return ref_img, img, has_Affine, False, l0_ref, linf_ref