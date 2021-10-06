import sys
sys.path.insert(0, "..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import trimesh
from nara.rasterizing import rasterizing
from nara.camera import Camera
import os
from os.path import join
from pathlib import Path
import numpy as np
import cv2
from torchvision.utils import save_image
import torchvision.transforms as T
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import time
import json

from dataset import Person, PersonDataset
from latent_codes import LatentCodes
from unet import UNet


from train import training
from test import test


device = torch.device("cuda:0") 
torch.manual_seed(42)

trans = T.Compose([T.ToTensor(), T.ToPILImage(), T.Resize(256),T.ToTensor()])
person_ids =[0,1,2]
Persons =[]
Datasets = []
for person_id in person_ids:
   person = Person(person_id)
   person_dataset = PersonDataset(person,transform=trans)
   print("person: ", person_id, "dataset length: ", len(person_dataset))
   Datasets.append(person_dataset)
   Persons.append(person)

batch_size = 1
data_set = ConcatDataset(Datasets)
data_loader = DataLoader(data_set,batch_size=batch_size,shuffle=True, num_workers=10,
                         pin_memory=False, drop_last=True)
total_frames = len(data_set)
print("Total : {0}, batch: {1}".format(len(data_set), len(data_loader)))

pid_dict = {0:"377",1: "386",2:"387"}
latent_dim = 16
training_path = "/home/user/output/training"  ## replik output dir
save_dir = join(training_path,"predictions")
Path(training_path).mkdir(parents=True, exist_ok=True)
Path(save_dir).mkdir(parents=True, exist_ok=True)
model_path= join(training_path,"model.pth")
epochs =50
person_all=["377","386","387"]

model= UNet(in_channels=19,out_channels=4).to(device)
optimizer = optim.Adam(model.parameters(),lr=0.002, betas=(0.9, 0.999))

#######     T R A I N        #######
load_trained_model= False

if(load_trained_model == False):
    model = training(model,data_loader,optimizer,epochs,model_path)
else:
    trained_model_path = join(training_path,"model_000009.pth")
    model = training(model,data_loader,optimizer,epochs,model_path,trained_model_path)

######      T E S T          #######
load_trained_model = False #True #False

if(load_trained_model==False):
    ### Test without loading saved model ###
    epoch_num = epochs-1
    test(model,optimizer,data_loader,save_dir=save_dir,epoch_num=epoch_num,train_path= training_path)
    #####################################################################
else:
    load_model_path= join(training_path,"sample_model_000049.pth")
    test(model,optimizer,data_loader,load_model_path, save_dir)


