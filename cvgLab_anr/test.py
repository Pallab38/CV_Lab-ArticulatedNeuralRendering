import os
import torch
import torch.optim as optim
from torchvision.utils import save_image
from os.path import join


from latent_codes import LatentCodes


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

def test(model,optimizer,data_loader,load_model_path=None,save_dir=None,epoch_num=-1,train_path=None):
    if load_model_path is not None:
        checkpoint = torch.load(load_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        i = checkpoint['epoch']
        
        training_path = os.path.dirname(load_model_path)
    else:
        assert epoch_num>-1,"Initialize the epoch number"
        i = epoch_num
        training_path = train_path
    model.eval()
    latent_dim=16
    person_all=["377","386","387"]
    with torch.no_grad():
        current_epoch = i -1
        latent_codes = LatentCodes(latent_dim, training_path, person_all, current_epoch, device)
        counter =0
        for data in data_loader:
            uv_image, normal_image,pid,gt_image,gt_mask = data
            uv_image = uv_image.permute(0,2,3,1).to(device) ## torch.Size([bs=1, 512, 512, 2])
            normal_image = normal_image.permute(0,2,3,1).to(device) ##torch.Size([bs=1, 512, 512, 3])
            latent, latent_optims = latent_codes.get_codes_for_training(list(pid)) # 256x256x16 latent variable PER PERSON
            latent = latent.permute(0,3,2,1) ## torch.Size([bs=1, 16, 256, 256])
            latent_img = torch.nn.functional.grid_sample(latent, uv_image,align_corners=True)
            latent_img = latent_img.permute(0,2,3,1) ##  torch.Size([1, 512, 512, 16])
            #input_img = torch.cat([normal_image, latent_img], axis=2) # (WxHx19)
            input_img = torch.cat([normal_image, latent_img], axis=3) # (bsxWxHx19) torch.Size([1, 512, 512, 19])

            input_img = input_img.permute(0,3,1,2)
            pred_img, pred_mask = model(input_img) ## torch.Size([1, 4, 512, 512])
            pred_img = pred_img.squeeze().detach().cpu() ## torch.Size([1, 3, 512, 512])
            pred_mask = pred_mask.squeeze().detach().cpu() ## torch.Size([512, 512])
           
            if(counter %100==0):
                save_img_path = join(save_dir, "pred_img_{:06d}.png".format(counter))
                save_mask_path = join(save_dir,"pred_mask_{:06d}.png".format(counter))
                save_image(pred_img, save_img_path)
                save_image(pred_mask,save_mask_path)
                
            counter +=1 






