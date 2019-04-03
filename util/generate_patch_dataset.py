import cv2
import numpy as np
import os
import h5py

####创建文件夹
def make_dir(Path1,Path2):
    Path = Path1
    if not os.path.exists(Path):
        os.mkdir(Path)
    Path = Path1+'/'+Path2
    if not os.path.exists(Path):
        os.mkdir(Path)
    return Path

#########通过H矩阵算左图到右图对应位置
def compute_corresponse(pt1,H_path):
    H=np.loadtxt(H_path,dtype=np.float32)
    pt2_list=[]
    for i in range(len(pt1)):
        (x1,y1)=pt1[i]
        x2=(H[0][0]*x1+H[0][1]*y1+H[0][2])/(H[2][0]*x1+H[2][1]*y1+H[2][2])
        y2=(H[1][0]*x1+H[1][1]*y1+H[1][2])/(H[2][0]*x1+H[2][1]*y1+H[2][2])
        pt2=(x2,y2)
        pt2_list.append(pt2)
    pt2_array=np.array(pt2_list)
    return pt2_array

######提取图像的sift特征点
def sift_detect(img,Max_kp_num):
    #change the image format to gray
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #extract sift features
    sift=cv2.xfeatures2d.SIFT_create(Max_kp_num)
    #detect the image keypoints
    keypoints=sift.detect(gray,None)
    return keypoints

def compute_valid_keypoints(Image_path,Patch_path,Max_kp_num):
    img=cv2.imread(Image_path)
    keypoints=sift_detect(img,Max_kp_num)
    #generate the pathes based on keypoints  
    valid_keypoints=generate_keypoints_patch(keypoints,Patch_path,img)
    print("the detect valid keypoints number "+str(len(valid_keypoints)))
    return valid_keypoints

def generate_keypoints_patch(keypoints,Patch_path,img):
    image_number = 0
    diff_keypoints_list = []
    for k in keypoints:
        #because the image axis is defferent from matrix axis
        x = int(k.pt[1])
        y = int(k.pt[0])
        patch_size = 32
        #judgment the boundary
        if (x-patch_size)>0 and (y-patch_size)>0 and (x+patch_size)<img.shape[0] and (y+patch_size)<img.shape[1] and (k.pt not in diff_keypoints_list):
            #delete the same keypoints
            diff_keypoints_list.append(k.pt)
            #the image of keypoint field
            img_patch = img[x-patch_size:x+patch_size,y-patch_size:y+patch_size]
            image_number = image_number+1
            patch_name = Patch_path+'/'+str(image_number)+'.jpg'
            cv2.imwrite(patch_name,img_patch)
            cv2.imshow('原始图像:',img_patch)
    #返回valid different keypoints
    return diff_keypoints_list

def save_correspond_patches(img,kp_pt,Patch_path):
    patch_size = 32
    image_number = 0
    for pt in kp_pt:
        image_number = image_number+1
        x = int(pt[1])
        y = int(pt[0])
        if (x-patch_size)>0 and (y-patch_size)>0 and (x+patch_size)<img.shape[0] and (y+patch_size)<img.shape[1]:
            img_patch = img[x-patch_size:x+patch_size,y-patch_size:y+patch_size]
            patch_name = Patch_path+'/'+str(image_number)+'.jpg'
            cv2.imwrite(patch_name,img_patch)

def save_patches(file_path):
    base_path = '/home/data1/daizhuang/patch_dataset/hpatches/hpatches_sequences_dataset/'+file_path+'/'
    Image_path_A = base_path+'1.ppm'
    Patch_path_A = make_dir('./dataset/patch_dataset_1000/'+file_path,str(1))
    print(Patch_path_A)
    pt1 = compute_valid_keypoints(Image_path_A,Patch_path_A,Max_kp_num=1000)
    for i in range(2,7):
        H_path = base_path+'H_1_'+str(i)
        Image_path_B = base_path+str(i)+'.ppm'
        Patch_path_B = make_dir('./dataset/patch_dataset_1000/'+file_path,str(i))
        img2 = cv2.imread(Image_path_B)
        pt2 = compute_corresponse(pt1,H_path)
        save_correspond_patches(img2,pt2,Patch_path_B)
        

if __name__ == "__main__":
    Image_data = "/home/data1/daizhuang/patch_dataset/hpatches/hpatches_sequences_dataset/"
    Count = 0
    for roots, dirs, files in os.walk(Image_data):
        for Dir in dirs:
            if Dir[0] == 'i':
                print('读取的图像:'+Dir)
                Count = Count+1
                print('读取的图片张数:'+str(Count))
                save_patches(Dir)
