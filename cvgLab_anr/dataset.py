import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import trimesh
from nara.rasterizing import rasterizing
from nara.camera import Camera
import os
from os.path import join
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from torchvision.utils import save_image
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as T
from tqdm import tqdm

pid_dict = {0:"377",1: "386",2:"387"}
def get_cameras_dict(path):
    extri_fpath = join(path,"extri.yml")
    intri_fpath = join(path,"intri.yml")
    extri_param = cv2.FileStorage(extri_fpath, flags=0)
    intri_param = cv2.FileStorage(intri_fpath, flags=0)
    n_cam = 23
    w,h = 1024,1024
    all_cameras = {}
    for i in range(n_cam):
        cam_num = "{:02d}".format(i+1)
        #print(cam_num)
        K = intri_param.getNode(f"K_{cam_num}").mat()
        dist = intri_param.getNode(f"dist_{cam_num}").mat()
        rvec = extri_param.getNode(f"R_{cam_num}").mat()  # 3x1 np.array
        tvec = extri_param.getNode(f"T_{cam_num}").mat()  # 3x1 np.array
        cam = Camera(rvec, tvec, K, dist, w, h)
        
        all_cameras["{:2d}".format(int(cam_num)).strip()]= cam

    return all_cameras

class Person:
    @staticmethod
    def get_all_person_ids(mesh_path="/home/user/easymocap/meshes"):
        return os.listdir(mesh_path)

    def __init__(self,person_id,root="/home/user/easymocap"):
        self.pid = pid_dict[person_id]
        self.mesh_dir = join(root,"meshes")
        self.p_path = join(self.mesh_dir,self.pid)
        self.frames = os.listdir(self.p_path)
        assert(len(self.frames)>0)
        self.img_root = join(root,"images")
        self.mask_root= join(root,"masks")
        self.frames_per_camera={}

        self.cam_path = join(root,"cameras",self.pid)
        #self.cameras = get_cameras(self.cam_path)
        self.cameras = get_cameras_dict(self.cam_path)
        #print(self.cameras_dict)
        #print(self.cameras)
        self.cid_all = list(self.cameras.keys())

    def get_num_frames(self):
        return len(os.listdir(self.p_path))
    
    def get_total_frames(self):
        return len(self.get_cid_frame_pairs())

    def get_frames(self):
        return self.frames
    
    def get_frames_for_camera(self,cid):
        if cid not in self.frames_per_camera:
            valid_frames =[]
            for frame in self.frames:
                valid_frames.append(frame)
            self.frames_per_camera[cid]=valid_frames
        # print("self.frames_per_camera[{0}] : {1}".format(cid,len(valid_frames)))
        # print("self.frames_per_camera: ",self.frames_per_camera)
        # print("type(self.frames_per_camera): ", type(self.frames_per_camera))
        return self.frames_per_camera
    
    def get_cid_frame_pairs(self):
        self._enumerate_cid_frame_pairs = []
        for cid in sorted(self.cid_all):
            frames_per_camera =  self.get_frames_for_camera(cid)
            # print(frames_per_camera[cid])
            for frame in frames_per_camera[cid]:
                self._enumerate_cid_frame_pairs.append((cid,frame))
        #print(self._enumerate_cid_frame_pairs)
        # print("len(self._enumerate_cid_frame_pairs): ",len(self._enumerate_cid_frame_pairs))
        return self._enumerate_cid_frame_pairs


class PersonDataset(Dataset):

    def __init__(self,person,transform=None):
        self.person = person
        self.uv_fpath = "../my_data/uv_table.npy"
        #self.uv_fpath = r"C:\Users\shunn0\Downloads\CV2_LAB\uv_table.npy"
        self.transformation= transform
    def __len__(self):
        # return self.person.get_num_frames()
        return self.person.get_total_frames()
    
    def __getitem__(self,idx):
        pid = self.person.pid
        frame, cid = self.get_frame_and_cid(idx)
        ##### get the video frames using pid and cid ####
        #c_id = "{:2d}".format(int(cid.split("_")[-1]))
        #print("c_id: ",c_id)
        #print("cid: ", cid)
        img_path = join(self.person.img_root,pid,cid,frame.replace("obj","jpg"))
        #print("img_path: ", img_path)
        mask_path = join(self.person.mask_root,pid,cid,frame.replace("obj","png"))
        #print("mask_path: ", mask_path)

        #img = Image.open(img_path)
        img = cv2.imread(img_path)
        #print("img.shape: ", img.shape)
        #print("type(img): ", type(img))
        #mask = Image.open(mask_path)
        mask = cv2.imread(mask_path,0)
        #print("mask.shape: ", mask.shape)
        #print("type(mask): ",type(mask))

        mesh_fpath = join(self.person.p_path,frame)
        cam = self.person.cameras[cid]
        uv_img,normal_img = self.get_uv_normal_image(mesh_fpath,cam)

        if self.transformation: 
            uv_img = self.transformation(uv_img)
            normal_img=self.transformation(normal_img)
            img = self.transformation(img)
            mask = self.transformation(mask)


        return uv_img,normal_img,pid,img,mask

    # def get_with_cam(self,index):
    #     frame, cid = self.get_frame_and_cid(idx)
    #     mesh_fpath = join(self.person.p_path,frame)
    #     cam = self.person.cameras[cid]
    #     uv_img,normal_img = self.get_uv_normal_image(mesh_fpath,cam)
    #
    #     return uv_img,normal_img

    def get_uv_normal_image(self, mesh_fpath,cam):
        mesh = trimesh.load(mesh_fpath)
        V = mesh.vertices
        F = mesh.faces
        T = np.load(self.uv_fpath)
        zbuffer, uv_image, normal_image = rasterizing(V, F, T, cam, calculate_normals=True)
        uv_image = (uv_image-0.5) * 2

        return uv_image, normal_image
    
    def get_frame_and_cid(self,idx):
        cid,frame = self.person.get_cid_frame_pairs()[idx]
        return frame,cid

if __name__ =="__main__":
    person_id=0
    person_1= Person(person_id)
    print("person_1.pid: ",person_1.pid) ## person_1.pid:  386
    print("person_1.get_num_frames(): ",person_1.get_num_frames()) ## person_1.get_num_frames():  646
    print("total: ", person_1.get_total_frames()) ## total:  14858
    idx = 12000
    cid, frame = person_1.get_cid_frame_pairs()[idx]
    print("idx: ",idx, " cid: ", cid, "  frame: ", frame) ## idx:  0  cid:  cam_01   frame:  000000.obj

    trans = T.Compose([T.ToTensor(), T.ToPILImage(), T.Resize(512),T.ToTensor()])
    #trans = T.Compose([T.ToTensor(), T.Resize(512),T.ToTensor()])

    p1_dataset = PersonDataset(person_1,transform=trans)
    print("length: ",len(p1_dataset))  ##length:  14858
    p1_data_loader = DataLoader(p1_dataset,batch_size=1)
    print("len(p1_data_loader): ",len(p1_data_loader))

    iter_count = 0
    for data in tqdm(p1_data_loader,desc="data"):
        # print("len(data): ",len(data))
        uv,normal,pid, gt_img,gt_mask = data
        print("uv.shape: ", uv.shape)
        print("normal.shape: ", normal.shape)
        print("pid: ", pid)
        print("gt_img.shape: ", gt_img.shape)
        print("gt_mask.shape: ", gt_mask.shape)
        
        print("gt_mask: ", torch.min(gt_mask.squeeze()).item() ,torch.max(gt_mask.squeeze()).item() )
        iter_count +=1
        print(uv.dtype, normal.dtype, gt_img.dtype, gt_mask.dtype)
        exit()
    print("Iteration in the data_loader: ", iter_count)

