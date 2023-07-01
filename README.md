# 2023-superai_SS3_level2
code 
## ก่อนทำการ train ใดๆ ก็ตามให้ไป load weight pretrain ด้วยนะครับ

https://github.com/ultralytics/assets/releases

[yolov8n-cls.pt](http://yolov8n-cls.pt)  —> weight สำหรับ classification

[yolov8n.pt](http://yolov8n.pt) —> weight สำหรับ object detection

## Detection
input : 1) รูป(ไม่ preprocess 256x256)  2) text file ที่ระบุ class , center_x, center_y, width_ratio, high_ratio
## Folder
ก่อนทำการ แบ่ง train : validation
intput
|--images 

|--labels

ใช้คำสั่ง 
splitfolders.ratio("./input", output="./input_split",seed=1337, ratio=(.8, .2), group_prefix=None, move=False)
และโยกย้ายโฟลเดอร์

หลังแบ่ง train : validation 80/20

train_split
|--train
   |--images
   |--labels

|--val
   |--images
   |--labels

## File ที่ต้องใช้
data.yaml

path: ../input_split  # dataset root dir
train: train/images  # train images (relative to 'path') 128 images
val: val/images  # val images (relative to 'path') 128 images
# test:  # test images (optional)

# Classes

names:
  0: Pulse
  1: Broad
  2: Narrow
