# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch

from .data_utils import read_image


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        if relabel:
            general_pid_set = set()
            general_cam_set = set()

            pid_set = {}
            cam_set = {}
            for i in img_items:
                general_pid_set.add(i[1])
                general_cam_set.add(i[2])

                pid=i[1]
                dataset=pid.split('_')[0]
                if dataset not in pid_set:
                    pid_set[dataset]=set()
                pid_set[dataset].add(i[1])
                
                camid=i[2]
                dataset=camid.split('_')[0]
                if dataset not in cam_set:
                    cam_set[dataset]=set()
                cam_set[dataset].add(i[2])

            self.pids = sorted(list(general_pid_set))
            self.cams = sorted(list(general_cam_set))
            self.pid_dict={}
            self.cam_dict={}
            for k,v in pid_set.items():
                pid_set[k]=sorted(list(v))
                if relabel:
                    self.pid_dict[k] = dict([(p, i) for i, p in enumerate(pid_set[k])])
            for k,v in cam_set.items():
                cam_set[k]=sorted(list(v))
                if relabel:
                    self.cam_dict[k] = dict([(p, i) for i, p in enumerate(cam_set[k])])
        
        else:
            pid_set = set()
            cam_set = set()
            for i in img_items:
                pid_set.add(i[1])
                cam_set.add(i[2])

            self.pids = sorted(list(pid_set))
            self.cams = sorted(list(cam_set))
            if relabel:
                self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
                self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])
    
    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)
        # mask_path=img_path.replace('/mnt/hdd1/liyuke','/home/nihao/wr/gaussresult').replace('/home/liyuke/data','/home/nihao/wr/gaussresult').split('.')[0]+'.pth'
        mask_path=img_path.replace('/ai/HAG/nihao/datasets','/ai/HAG/nihao/gaussresult').split('.')[0]+'.pth'
        # mask=Image.open(mask_path)
        # mask=mask.resize((img.shape[2],img.shape[1]))
        # mask=np.array(mask)
        # mask=torch.tensor(mask,dtype=torch.uint8)#h,w
        # mask[mask==0]=int(i*255)
        # img*=mask
        # img/=255
        mask=torch.load(mask_path)
        if self.relabel:
            pid = self.pid_dict[pid.split('_')[0]][pid]
            camid = self.cam_dict[camid.split('_')[0]][camid]
        return {
            "images": img,
            "targets": pid,
            "camids": camid,
            "img_paths": img_path,
            "masks":mask
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)
