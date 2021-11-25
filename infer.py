import torch
import os
import cv2
import time
from tqdm import tqdm
from google_drive_downloader import GoogleDriveDownloader as gdd
from models.models import *
from utils.datasets import letterbox
from utils.general import non_max_suppression, xyxy2xywh
from torchvision import transforms
import json

def load_image(img_path, imgsz):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = cv2.imread(img_path)  # BGR
    h0, w0 = img.shape[:2]  # orig hw
    r = imgsz / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        
    return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

    
checkpoint = "./runs/train/vrdl_hw_pretrained7/weights/best_ap.pt"
# checkpoint = "./yolor_p6.pt"
device = 'cuda:0'
imgsz = 640
model = Darknet('cfg/yolor_p6.cfg').to(device)
try:
    ckpt = torch.load(checkpoint, map_location=device)  # load checkpoint
    ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(ckpt['model'], strict=False)
except:
    load_darknet_weights(model, checkpoint)
imgsz = check_img_size(imgsz, s=64)  # check img_size

model.eval()
data_listdir = os.listdir("./test")
print(len(data_listdir))

stride = 64
TEST_IMAGE_NUMBER = 100 # This number is fixed.
test_img_list = []

# Read image (Be careful with the image order)
data_listdir.sort(key = lambda x: int(x[:-4]))

for img_name in data_listdir[:TEST_IMAGE_NUMBER]:
    img_path = os.path.join("./test", img_name)
    img, (h0, w0), (h, w) = load_image(img_path, imgsz)
    
    nh = stride*((h//stride)+1) if h % stride != 0 else h
    nw = stride*((w//stride)+1) if w % stride != 0 else w
    img, ratio, pad = letterbox(img, (nh, nw), auto=False)
    shapes = (h0, w0), ((h / h0, w / w0), pad)

    test_img_list.append(img)

conf_thres = 0.001
iou_thres = 0.65

start_time = time.time()
for img in tqdm(test_img_list):
    img = transforms.ToTensor()(img).unsqueeze(0).to(device)
    import pdb
    pdb.set_trace()
    inf_out, train_out = model(img)
    output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres)
end_time  = time.time()    
print("\nInference time per image: ", (end_time - start_time) / len(test_img_list))
jdict = []
for img_name in tqdm(data_listdir):
    img_path = os.path.join("./test", img_name)
    img, (h0, w0), (h, w) = load_image(img_path, imgsz)
    
    nh = stride*((h//stride)+1) if h % stride != 0 else h
    nw = stride*((w//stride)+1) if w % stride != 0 else w
    img, ratio, pad = letterbox(img, (nh, nw), auto=False)
    shapes = (h0, w0), ((h / h0, w / w0), pad)

    img = transforms.ToTensor()(img).unsqueeze(0).to(device)
    inf_out, train_out = model(img)
    pred = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres)[0]    
    
    box = pred[:, :4].clone()
    
    scale_coords(img.shape[1:], box, shapes[0], shapes[1])
    box = xyxy2xywh(box)
    box[:, :2] -= box[:, 2:] / 2
    
    for p, b in zip(pred.tolist(), box.tolist()):
        jdict.append({'image_id': int(img_name.split('.')[0]),
                      'category_id': int(p[5]),
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})        
    
pred_json = "answer.json"
with open(pred_json, 'w') as f:
    json.dump(jdict, f)

    