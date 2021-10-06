#import sys
#sys.path.insert(0, "..")
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
from tqdm import tqdm
import time
import json

from latent_codes import LatentCodes



device = torch.device("cuda:0") 
torch.manual_seed(42)

def training(model, data_loader, optimizer, epochs, model_path, load_model_path=None):
    training_path = os.path.dirname(model_path)
    latent_dim =16
    person_all=["377","386","387"]
    if load_model_path is not None:
        print("...  Loading Trained Model  ...")
        checkpoint = torch.load(load_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']+1
    else: 
        start_epoch=0
    l1_loss = nn.L1Loss()
    bce_loss = nn.BCELoss()
    loss_dict={}
    model.train()
    for i in tqdm(range(start_epoch,epochs),desc="epochs"):
        print("Training started, current epoch: ",i)
        current_epoch = i -1
        latent_codes = LatentCodes(latent_dim, training_path, person_all, current_epoch, device)
        counter=0
        running_loss=0
        for data in tqdm(data_loader,desc="Batch"):
            optimizer.zero_grad()
            uv_image, normal_image,pid,gt_image,gt_mask = data
            uv_image = uv_image.permute(0,2,3,1).to(device) ## torch.Size([bs=1, 512, 512, 2])
            normal_image = normal_image.permute(0,2,3,1).to(device) ##torch.Size([bs=1, 512, 512, 3])
            gt_image = gt_image.to(device) ##  torch.Size([bs=1, 3, 512, 512])
            gt_mask =  gt_mask.squeeze().to(device) ## torch.Size([512, 512])->torch.Size([bs=1,512, 512])
            latent, latent_optims = latent_codes.get_codes_for_training(list(pid)) # 256x256x16 latent variable PER PERSON
            latent = latent.permute(0,3,2,1) ## torch.Size([bs=1, 16, 256, 256])
            latent_img = torch.nn.functional.grid_sample(latent, uv_image,align_corners=True)
            latent_img = latent_img.permute(0,2,3,1) ##  torch.Size([1, 512, 512, 16])
            #input_img = torch.cat([normal_image, latent_img], axis=2) # (WxHx19)
            input_img = torch.cat([normal_image, latent_img], axis=3) # (bsxWxHx19) torch.Size([1, 512, 512, 19])
        
            input_img = input_img.permute(0,3,1,2)
            pred_img,pred_mask = model(input_img) ## torch.Size([1, 4, 512, 512])
            
            pred_mask = pred_mask.squeeze()
            loss_img = l1_loss(pred_img,gt_image)
            loss_mask = bce_loss(pred_mask, gt_mask)
            total_loss = loss_img  + loss_mask
            #print("total loss: ",total_loss.item())
            running_loss += total_loss.item() * uv_image.shape[0]
            total_loss.backward()
            optimizer.step()

            
        latent_codes.save_all_codes(i)
        
        save_model_path = join(training_path,"model_{:06d}.pth".format(i))
        torch.save({'epoch': i,
                    'model_state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict()}, 
                   save_model_path
                   )
        epoch_loss = running_loss / len(data_loader)

        print("Epoch: {}, Loss: {:.04f}".format(i,epoch_loss))
        loss_dict[i] = epoch_loss  

        fpath = join(training_path,"loss.txt")
        with open(fpath,"w") as f_loss:
            json.dump(loss_dict,f_loss)

    print("saving directory: ", training_path)

    
    return model        

