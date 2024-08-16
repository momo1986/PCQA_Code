from PIL import Image
import cv2
import requests
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import time
from multiprocessing.dummy import Pool as ThreadPool
class IQADataset:
    def __init__(self, images, base_path='../data/train/train/', modelcfg=None, aug=0 ):
        
        self.images = images.copy()
        self.base_path = base_path
        self.transform = create_transform(**modelcfg)
        self.aug=aug
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, item):
        img = Image.open(self.base_path + self.images[item]).convert('RGB')
        img = self.transform(img)
        return img
if 1:
    names = ['tf_efficientnet_b0', 'cait_m48_448', 'eca_nfnet_l1',"gluon_seresnext101_64x4d",'swin_large_patch4_window12_384_in22k', 'swin_base_patch4_window12_384','tf_efficientnet_b7_ns','tf_efficientnetv2_l_in21ft1k','vit_deit_base_distilled_patch16_384']
   
    train = pd.read_excel('../dataimg/train.xlsx')
    val = pd.read_excel('../dataimg/val.xlsx')
    print(train)
    
    for arch in names:
        
        starttime = time.time()

        model = timm.create_model(arch, pretrained=True).to('cuda')
        model.eval()

        train_dataset = IQADataset(
            images = train.name.values,
            base_path='../dataimg/train/train/',
            modelcfg = resolve_data_config({}, model=model),
            aug = 0,
        )
        val_dataset = IQADataset(
            images = val.name.values,
            base_path='../dataimg/val/val/',
            modelcfg = resolve_data_config({}, model=model),
            aug = 0,
        )
        BS = 16
        train_dataloader = DataLoader(train_dataset, batch_size=BS, num_workers= 4, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=BS, num_workers= 4, shuffle=False)
       
        with torch.no_grad():
            res = [model(img.to('cuda')).cpu().numpy() for img in tqdm(train_dataloader)]
            res = np.concatenate(res, 0)
            pqts ={'image_feature':[]}
            for result in res:
                pqts['image_feature'].append(list(result))
            pqts = pd.DataFrame(pqts)
            pqts['name'] = train['name']
            pqts['prompt']=train['prompt']
            pqts['mos'] = train['mos']
            pqts.to_parquet(arch.replace('/','.') + '_train.parquet',index=False)
            res = [model(img.to('cuda')).cpu().numpy() for img in tqdm(val_dataloader)]
            res = np.concatenate(res, 0)
            pqts ={'image_feature':[]}
            for result in res:
                pqts['image_feature'].append(list(result))
            pqts = pd.DataFrame(pqts)
            pqts['name'] = val['name']
            pqts['prompt']=val['prompt']
            pqts['mos'] = 0
            pqts.to_parquet(arch.replace('/','.') + '_val.parquet',index=False)

        print( arch, ', Done in:', int(time.time() - starttime), 's' )
    