import numpy as np
import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
###复制文件的一个包
import shutil

loader = transforms.Compose([transforms.ToTensor()]) 

def image_loader(image_name): 
    image = Image.open(image_name).convert('RGB') 
    image = loader(image).unsqueeze(0) 
    return image.to(device, torch.float)

def countFile(dir): 
    tmp = 0 
    for item in os.listdir(dir): 
        if os.path.isfile(os.path.join(dir, item)): 
            tmp += 1 
        else: 
            tmp += countFile(os.path.join(dir, item)) 
    return tmp

def find_files(file_dir): 
    global image_number
    for i in range(2,7):
        patch_image_path_1 = os.path.join(file_dir,str(1))
        patch_image_path = os.path.join(file_dir,str(i))

        left_patch = os.path.join("./dataset/same_patch_dataset_1000", 'left')
        right_patch = os.path.join("./dataset/same_patch_dataset_1000",'right')

        for item_1 in os.listdir(patch_image_path_1):
            if os.path.isfile(os.path.join(patch_image_path,item_1)):
                image_number = image_number+1
                left_patch_file = os.path.join(left_patch, str(image_number)+'.jpg')
                right_patch_file = os.path.join(right_patch, str(image_number)+'.jpg')
                shutil.copyfile(os.path.join(patch_image_path_1, item_1), left_patch_file)
                shutil.copyfile(os.path.join(patch_image_path, item_1), right_patch_file)
            else:
                print("不存在的图片:"+os.path.join(patch_image_path,item_1))
      
if __name__ == "__main__":
    Patch_data = "./dataset/patch_dataset_1000/"
    Count = 0
    image_number = 0 
    for roots, dirs, files in os.walk(Patch_data):
        for files in dirs:
            if files[0] == 'i':
                print("读取的图像文件夹:" + files)   
                Count = Count+1
                print("读取的图像张数:" +str(Count)) 
                full_files_path = Patch_data + files
                find_files(full_files_path)
    print('总共图像张数:' + str(image_number))
    print("finished")
