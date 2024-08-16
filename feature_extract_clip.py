from PIL import Image
import cv2
import requests
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel

from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
# for model_name in  [ 'openai/clip-vit-large-patch14-336', 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',"laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"]:
for model_name in  [ 'openai/clip-vit-base-patch32', 'laion/CLIP-ViT-L-14-laion2B-s32B-b82K']:
    model = CLIPModel.from_pretrained(model_name) 
    processor = CLIPProcessor.from_pretrained(model_name)
    model.cuda().eval()
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    def infer_one(images, text="a photo of a cat"):
        inputs = processor(text, images=images, return_tensors="pt", padding=True, truncation = True).to('cuda')
        outputs = model(**inputs)
        emb_image = outputs.image_embeds # this is the image-text similarity score
        emb_text = outputs.text_embeds
        return [emb_image,emb_text]

    train = pd.read_excel('../dataimg/train.xlsx')
    print(train)
    pqts ={'image_feature':[],'text_feature':[]}
    p=ThreadPool(4)
    def func(i):
        if train.iloc[i]['name'][-3:] == 'png':
            images = [Image.open('../dataimg/train/train/'+ train.iloc[i]['name'])]
        else:
            clip = video = cv2.VideoCapture('../data/train/train/'+ train.iloc[i]['name'])
            frame_count = 0 # 记录已处理的帧数
            success = True # 判断是否还有未处理完的帧
            images = []
            while success:
                success, frame = video.read()
                if not success:break
                images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            
        with torch.no_grad():
            result = infer_one(images, train.iloc[i]['name'][:-8] + train.iloc[i].prompt)
        return result
    results = p.imap(func,[i for i in range(len(train))])
    for result in tqdm(results,total = len(train)):
        pqts['image_feature'].append(list(result[0].cpu().numpy()))
        pqts['text_feature'].append(list(result[1].cpu().numpy()))
    pqts = pd.DataFrame(pqts)
    pqts['name'] = train['name']
    pqts['prompt']=train['prompt']
    pqts['mos'] = train['mos']
    pqts.to_parquet(model_name.replace('/','.') + '_train.parquet',index=False)
    val = pd.read_excel('../dataimg/val.xlsx')
    pqts ={'image_feature':[],'text_feature':[]}
    def func(i):
        if val.iloc[i]['name'][-3:] == 'png':
            images = [Image.open('../dataimg/val/val/'+ val.iloc[i]['name'])]
        else:
            clip = video = cv2.VideoCapture('../dataimg/val/val/'+ val.iloc[i]['name'])
            frame_count = 0 # 记录已处理的帧数
            success = True # 判断是否还有未处理完的帧
            images = []
            while success:
                success, frame = video.read()
                if not success:break
                images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            
        with torch.no_grad():
            result = infer_one(images, val.iloc[i]['name'][:-8] +val.iloc[i].prompt)
        return result
    results = p.imap(func,[i for i in range(len(val))])
    for result in tqdm(results,total = len(val)):
        pqts['image_feature'].append(list(result[0].cpu().numpy()))
        pqts['text_feature'].append(list(result[1].cpu().numpy()))
    pqts = pd.DataFrame(pqts)
    pqts['name'] = val['name']
    pqts['prompt']=val['prompt']
    pqts['mos'] = 0
    pqts.to_parquet(model_name.replace('/','.') + '_val.parquet',index=False)