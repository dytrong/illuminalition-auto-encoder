import torch.nn as nn
import torchvision.models as models  
import torch
import torchvision.transforms as transforms
from PIL import Image
from compute_distance import compute_cross_correlation_match
from compute_average_precision import compute_AP
from compute_keypoints_patch import * 
import time
import sys
import configparser
import argparse
from sklearn import preprocessing

#####global variable######
img_to_tensor = transforms.ToTensor()

######初始化参数
config = configparser.ConfigParser()
config.read('./setup.ini')
Max_kp_num = config.getint("DEFAULT", "Max_kp_num")
img_suffix = config.get("DEFAULT", "img_suffix")
txt_suffix = config.get("DEFAULT", "file_suffic")
Model_Img_size = config.getint("DEFAULT", "Model_Image_Size")


######接收参数
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",type=str,choices=['i','v'],required=False)
parser.add_argument("--model_parameters_path", type=str, required=False)
args = parser.parse_args()


#####change images to tensor#####
def change_images_to_tensor(H5_Patch, norm_flag = True):
    img_list=[]
    #the patch image .h5 file
    Img_h5=h5py.File(H5_Patch,'r')
    for i in range(len(Img_h5)):
        img=Img_h5[str(i)][:]
        ###change image format from cv2 to Image
        img=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        img=img.resize((Model_Img_size,Model_Img_size))
        img=img_to_tensor(img)
        img=img.numpy()
        img=img.reshape(3,Model_Img_size,Model_Img_size)
        img_list.append(img)
    img_array=np.array(img_list)
    #####将数据转为tensor类型
    img_tensor=torch.from_numpy(img_array)
    #####数据归一化
    if norm_flag:
        img_tensor -= 0.424
        img_tensor /= 0.270
    return img_tensor

def compute_batch_descriptors(Img_path,H5_Patch):
    ####generate patch img .h5 file and return valid key points
    valid_keypoints=compute_valid_keypoints(Img_path,H5_Patch,Max_kp_num)
    #####H5_patch_path用来保存每张图片的keypoint patches,方便计算CNN描述符
    input_data=change_images_to_tensor(H5_Patch)
    #####计算AutoEncoder描述符
    sys.path.append("../util") 
    from autoencoder_descriptors import generate_autoencoder_desc
    desc = generate_autoencoder_desc(input_data, args.model_parameters_path)
    return valid_keypoints,desc

def compute_mAP(file_path):
    total_AP = []
    extract_desc_time = []
    compute_desc_dis_time = []
    for i in range(2,7):
        print("start compute the "+str(i-1)+" pairs matches")
        base_path='/home/data1/daizhuang/patch_dataset/hpatches/hpatches_sequences_dataset/'+str(file_path)+'/'
        ####ground truth of Homography 
        H_path = base_path+'H_1_'+str(i)
        Img_path_A = base_path+str(1)+img_suffix
        Img_path_B = base_path+str(i)+img_suffix
        ######patches path
        H5_Patch_A = './data/h5_patch/img'+str(1)+txt_suffix
        H5_Patch_B = './data/h5_patch/img'+str(i)+txt_suffix
        ######read image
        img1 = cv2.imread(Img_path_A)
        img2 = cv2.imread(Img_path_B)
        #############提取特征点，和卷积描述符
        start = time.time()
        kp1, desc1 = compute_batch_descriptors(Img_path_A, H5_Patch_A)
        kp2, desc2 = compute_batch_descriptors(Img_path_B, H5_Patch_B)
        extract_desc_time.append(time.time()-start)
        ##############计算描述符之间的距离并寻找特征点匹配对
        start = time.time()
        match_cc = compute_cross_correlation_match('cos', des1=desc1, des2=desc2)
        compute_desc_dis_time.append(time.time()-start)
        ##############compute average precision(AP)
        AP = compute_AP(img1,img2,kp1,kp2,match_cc,H_path,'cos')
        total_AP.append(AP)
    mAP = np.mean(total_AP)
    print('提取描述符平均耗时:'+str(np.mean(extract_desc_time)))
    print('计算描述符距离平均耗时:'+str(np.mean(compute_desc_dis_time)))
    return mAP

if __name__ == "__main__":
    start = time.time()
    all_mAP = []
    Count = 0
    Image_data = "/home/data1/daizhuang/patch_dataset/hpatches/hpatches_sequences_dataset/"
    for roots, dirs, files in os.walk(Image_data):
        #####tqdm显示进度条
        for DIR in dirs:
            if DIR[0] == args.dataset:
                print('读取的图像:'+DIR)
                Count = Count+1
                print('读取的图片张数:'+str(Count))
                mAP =compute_mAP(DIR)
                all_mAP.append(mAP)
                print('\n')
    print('所有数据的平均精度为:'+str(np.sum(all_mAP)/len(all_mAP)))
    end=time.time()
    print('总共耗时:'+str(end-start))           
