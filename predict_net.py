from yolo_top import yolov3
import numpy as np
import tensorflow as tf
import config as cfg
from PIL import Image, ImageDraw, ImageFont
from draw_boxes import draw_boxes
import matplotlib.pyplot as plt
import cv2
import os
import random
import csv
def nms(boxes,scores, threshold, method):
    if boxes.size==0:
        return np.empty((0,3))
    x1 = boxes[:,1]
    y1 = boxes[:,0]
    x2 = boxes[:,3]
    y2 = boxes[:,2]
    s = scores[:,0]
    area = (x2-x1+1) * (y2-y1+1)
    area_2=[]
    I = np.argsort(area)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size>0:
        i = I[-1]
        pick[counter] = i
        area_2.append(area[i])
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w * h
        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o<=threshold)]
    pick = pick[0:counter]
    area_2=np.reshape(area_2,[-1,1])
    mean_area=np.mean(area_2)
    return pick,mean_area

sliceHeight=700 
sliceWidth=700
overlap=0.4
verbose=False
name_file=open('E:/Python/tensorflow/count/test.txt','r')
name_file=name_file.readlines()              
imgs_holder = tf.placeholder(tf.float32, shape=[1, 416, 416, 3])
istraining = tf.constant(False, tf.bool)
#img_dir='E:/Python/tensorflow/count/test_dataset_new/'
#pic_list=os.listdir(img_dir)
#random.shuffle(pic_list)
model = yolov3(imgs_holder, None, istraining)
img_hw = tf.placeholder(dtype=tf.float32, shape=[2])
boxes, scores = model.pedict(img_hw, iou_threshold=0.5, score_threshold=0.7)
out = open('E:/Python/tensorflow/count/Stu_csv.csv','w', newline='')
csv_write = csv.writer(out)
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'E:/Python/tensorflow/count/models/yolov3.ckpt-80000')  
    pic_num=0     
    for name in name_file:
        image=[]
        name=name[:-1]
        image_path='E:/Python/tensorflow/count/test_dataset/'+name
        image1 = cv2.imread(image_path)  # color
        image0 = np.array(image1, dtype='float32')/255
        win_h, win_w = image0.shape[:2]   
        pad = 0
        if sliceHeight > win_h:
            pad = sliceHeight - win_h
        if sliceWidth > win_w:
            pad = max(pad, sliceWidth - win_w)
        # pad the edge of the image with black pixels
        if pad > 0:    
            border_color = (0,0,0)
            image0 = cv2.copyMakeBorder(image0, pad, pad, pad, pad, 
                                     cv2.BORDER_CONSTANT, value=border_color)
    
        dx = int((1. - overlap) * sliceWidth)
        dy = int((1. - overlap) * sliceHeight)
        Total_box=[] 
        Total_seores=[]
        for y0 in range(0, image0.shape[0], dy):#sliceHeight):
            for x0 in range(0, image0.shape[1], dx):#sliceWidth):
                # make sure we don't have a tiny image on the edge
                image0 = cv2.imread(image_path)  # color
                image0 = np.array(image0, dtype='float32')/255
                if y0+sliceHeight > image0.shape[0]:
                    y = image0.shape[0] - sliceHeight
                else:
                    y = y0
                if x0+sliceWidth > image0.shape[1]:
                    x = image0.shape[1] - sliceWidth
                else:
                    x = x0
                window_c=image0[y:y + sliceHeight, x:x + sliceWidth]
    #            cv2.imwrite('E:/Python/tensorflow/count/test_dataset_new/'+str(j)+'.jpg',window_c)
    #            j=j+1
                image_data=cv2.resize(window_c,(416,416))
                boxes_, scores_ = sess.run([boxes, scores],
                                                     feed_dict={
                                                                img_hw:[sliceHeight, sliceHeight],
                                                                imgs_holder: np.reshape(image_data , [1, 416, 416, 3])})         
                num_box=len(boxes_)
                for j in range(num_box):
                    if (boxes_[j,2]-boxes_[j,0])/(boxes_[j,3]-boxes_[j,1])>2 or (boxes_[j,2]-boxes_[j,0])/(boxes_[j,3]-boxes_[j,1])<0.5:
                        continue
                    Total_box.append(int(boxes_[j,0]+y))
                    Total_box.append(int(boxes_[j,1]+x))
                    Total_box.append(int(boxes_[j,2]+y))
                    Total_box.append(int(boxes_[j,3]+x))
                    Total_seores.append(scores_[j])
        Total_box1=np.reshape(Total_box,[-1,4])
        Total_seores1=np.reshape(Total_seores,[-1,1])
        pick,mean_area=nms(Total_box1,Total_seores1,0.5,'Min')
        total_box_num=len(pick)
        for k in range(total_box_num):
            num=pick[k]
            bb=[]
            if ((Total_box1[num,3]-Total_box1[num,1])*(Total_box1[num,2]-Total_box1[num,0]))<(mean_area*0.5):
                continue             
            bb.append(str(Total_box1[num,1]))
            bb.append(str(Total_box1[num,0]))
            bb.append(str(Total_box1[num,3]))
            bb.append(str(Total_box1[num,2]))
            bb1=" ".join(bb)
            stu=[name,bb1]
            csv_write.writerow(stu)
            cv2.rectangle(image1, (Total_box1[num,1],Total_box1[num,0]), (Total_box1[num,3],Total_box1[num,2]), (0,255,0), 2)
        cv2.imwrite('E:/Python/tensorflow/count/val_dataset_1/'+str(pic_num)+'.jpg',image1)
#        cv2.namedWindow('image',cv2.WINDOW_NORMAL) 
#        cv2.imshow('image',image0)
#        cv2.waitKey(0) 
        pic_num=pic_num+1

