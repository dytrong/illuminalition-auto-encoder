import cv2
import numpy as np
import h5py
import math
import time
from matplotlib import pyplot as plt
import torch
from sklearn import preprocessing
from compute_distance import compute_dis_matrix


##############计算像素的欧式距离######################################
def compute_pixs_distance(des1,des2):
    L2_dis=compute_dis_matrix(des1,des2,'L2')
    Max_Matches=np.dtype({'names':['i','j','MAX'],'formats':['i','i','f']})
    ###初始化保存从左到右最佳匹配的矩阵
    M1=np.zeros(L2_dis.shape[0],dtype=Max_Matches)
    MAX_XY=[]
    #compute the max match from left to right
    for i in range(L2_dis.shape[0]):
        MIN_L2_DIS=np.min(L2_dis[i,:]) 
        M1[i]['i']=i
        M1[i]['j']=np.argmin(L2_dis[i,:]) 
        M1[i]['MAX']=MIN_L2_DIS
        MAX_XY.append([M1[i]['i'],M1[i]['j'],M1[i]['MAX']])
    MAX_XY=np.array(MAX_XY)
    return MAX_XY

####################################################################
########################计算ground true#############################
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

#########计算两幅图像中实际匹配点对
def points_distance(pt1,pt2,H_path):
    #####计算图1经过H变换后在图2中的估计位置
    pt1=compute_corresponse(pt1,H_path)
    pt1=np.array(pt1)
    pt2=np.array(pt2)
    #####计算图1估计出的位置与图二实际位置的欧式距离
    L2_dis=compute_pixs_distance(pt1,pt2) 
    valid_match=[]
    #####找出实际真正的匹配对
    for i in range(len(L2_dis)):
        ####当欧式距离小于1.5像素，认为是同一个点
        if L2_dis[i][2]<=3:
            valid_match.append(L2_dis[i])  
    valid_match=np.array(valid_match)     
    return valid_match 

def compute_valid_match(max_match,Thresh,method):
    valid_match=[]
    for i in range(len(max_match)):
        ######当距离公式选用L2范数时，阈值是小于等于
        if max_match[i][2]<=Thresh and method=='L2':
            valid_match.append(max_match[i])
        #####当距离公式选用cos距离时,阈值是大于等于
        if max_match[i][2]>=Thresh and method=='cos':
            valid_match.append(max_match[i])
    valid_match=np.array(valid_match)
    return valid_match

def compute_PR(max_match,thresh,ground_true_match,imshow,method):
    max_match=compute_valid_match(max_match,thresh,method)
    correct_match_number=0
    correct_match=[]
    detect_match_number=len(max_match)
    ground_true_match_number=len(ground_true_match)
    if ground_true_match_number==0 or len(max_match)==0:
        return 0,0 
    for i in range(detect_match_number):
        ####看检测出的匹配点是否在ground_true_match中,若在则返回它的位置
        if max_match[i][0] in list(ground_true_match[:,0]):
            #####numpy.array没有index这个属性
            index=list(ground_true_match[:,0]).index(max_match[i][0])
            if max_match[i][1]==ground_true_match[index][1]:
                 correct_match_number=correct_match_number+1
                 correct_match.append(max_match[i])
    if detect_match_number==0:
        return 0,0
    recall=float(correct_match_number)/ground_true_match_number
    precision=float(correct_match_number)/detect_match_number
    if imshow:
        return precision,recall,correct_match,max_match
    else:
        return precision,recall

def compute_AP(img1,img2,kp1,kp2,max_match,H_path,method,imshow=False):
    if imshow:
        #####显示匹配对
        ground_true_match=points_distance(kp1,kp2,H_path)
        if len(max_match)==0:
            print('不存在匹配对')
            return 0
        max_list=max_match[:,2]
        max_list.sort()
        if method=='L2':
            Thresh=max_list[29]
        if method=='cos':
            Thresh=max_list[-30]
        #######max_match为算法检测出的匹配对,
        #######correct_match为检测出的算法为真实的匹配对
        p,r,correct_match,max_match=compute_PR(max_match,Thresh,ground_true_match,imshow,method)
        print('正确率为:'+str(p))
        show_keypoints(kp1,kp2,img1,img2,correct_match,max_match) 
    else:
        ground_true_match=points_distance(kp1,kp2,H_path) ####实际匹配对
        max_number=np.max(max_match[:,2])
        min_number=np.min(max_match[:,2])
        step_number=50
        P_list=[]
        R_list=[]
        AP=[]
        for i in range(step_number):
            thresh=max_number-((max_number-min_number)/step_number)*i
            precision,recall=compute_PR(max_match,thresh,ground_true_match,imshow,method)
            P_list.append(precision)
            R_list.append(recall)
        #####当选用L2范数时，计算出的精度是小到大的，
        #####这样我们算PR面积的时候公式就和cos算出的不统一了
        #####所以将PR list反转一下，和cos计算出来的就一致了
        if method=='L2':
            P_list.reverse() ###将列表反转
            R_list.reverse() 
        for i in range(1,step_number):
            AP.append(P_list[i]*(R_list[i]-R_list[i-1]))
        average_AP=np.sum(AP)
        print("The Average Precision of a pair images:"+str(average_AP))
        return average_AP

def show_keypoints(kp1,kp2,img1,img2,detect_correct_match,max_match):
    cols1 = img1.shape[1]
    img3=append_image(img1,img2)
    if len(max_match)==0:
        return
    key_points1=[]
    key_points2=[]
    kp_1=[]
    kp_2=[]
    #####画所有算法认为是匹配上的点
    for i in range(len(max_match)):
        kp_1.append(kp1[int(max_match[i][0])])
        kp_2.append(kp2[int(max_match[i][1])])
    #####ground truth
    for i in range(len(detect_correct_match)):
        key_points1.append(kp1[int(detect_correct_match[i][0])])
        key_points2.append(kp2[int(detect_correct_match[i][1])]) 
    for i in range(len(kp_1)):
        (x1,y1)=kp_1[i]
        (x2,y2)=kp_2[i]
        cv2.circle(img3,(int(np.round(x1)),int(np.round(y1))),2,(0,255,255),2)
        cv2.circle(img3,(int(np.round(x2)+cols1),int(np.round(y2))),2,(0,155,255),2)
        if (kp_1[i] in key_points1) and (kp_2[i] in key_points2):
            cv2.line(img3, (int(np.round(x1)),int(np.round(y1))), (int(np.round(x2)+cols1),int(np.round(y2))), (255, 0, 0), 2, lineType=cv2.LINE_AA, shift=0)
        else:
            cv2.line(img3, (int(np.round(x1)),int(np.round(y1))), (int(np.round(x2)+cols1),int(np.round(y2))), (0, 0, 255), 2, lineType=cv2.LINE_AA, shift=0)
    cv2.imwrite('/home/daizhuang/example.jpg',img3)
    cv2.imshow("img3",img3)
    cv2.waitKey(0)

#################################
#####图像拼接#####
def append_image(img1,img2):
  rows1=img1.shape[0]
  rows2=img2.shape[0]
  if rows1<rows2:
      concat=np.zeros((rows2-rows1,img1.shape[1],3),dtype=np.uint8)
      img1=np.concatenate((img1,concat),axis=0)
  if rows1>rows2:
      concat=np.zeros((rows1-rows2,img2.shape[1],3),dtype=np.uint8)
      img2=np.concatenate((img2,concat),axis=0)
  img3=np.concatenate((img1,img2), axis=1)
  return img3
