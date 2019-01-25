import numpy as np
import os
import pandas as pd
import tensorflow as tf
from PIL import Image
import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg    
import math  
if __name__ == "__main__":
    date_dir='E:/Python/tensorflow/count'
#    tf_filename='trainval_2018.tfrecord'
    max_box=500
    sliceHeight=700 
    sliceWidth=700
    overlap=0.2
    verbose=False
#    tf_filename = os.path.join(date_dir,tf_filename)
#    writer = tf.python_io.TFRecordWriter(tf_filename)   
    #data = pd.read_table("E:/Python/tensorflow/count/YOLO3/Stu_csv.csv",sep=",")
    data = pd.read_table("E:/Python/tensorflow/count/train_labels.csv",sep=",")
    name_file=open('E:/Python/tensorflow/count/train.txt','r')
    name_file=name_file.readlines()
    j=1
    for name in name_file:
        name=name[:-1]
        out_name=name.split('.')[0]
        image_path='E:/Python/tensorflow/count/train_dataset/'+name
        outdir='E:/Python/tensorflow/count/train_dataset_new/'
        xx = np.array(data[data['ID'] == name][' Detection'])
        image0 = cv2.imread(image_path)  # color
        image0 = np.array(image0, dtype='float32')/255
        win_h, win_w = image0.shape[:2]   
        bbox=np.zeros((max_box,4), dtype=np.float32)
        num_box=xx.shape[0]
        for i in range(xx.shape[0]):
            bbox[i,0]=(int((xx[i].split(' '))[0]))
            bbox[i,1]=(int((xx[i].split(' '))[1]))
            bbox[i,2]=(int((xx[i].split(' '))[2]))
            bbox[i,3]=(int((xx[i].split(' '))[3]))
        # if slice sizes are large than image, pad the edges
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
    
        n_ims = 0
        dx = int((1. - overlap) * sliceWidth)
        dy = int((1. - overlap) * sliceHeight)
    
        for y0 in range(0, image0.shape[0], dy):#sliceHeight):
            for x0 in range(0, image0.shape[1], dx):#sliceWidth):
                n_ims += 1
#                image0 = cv2.imread(image_path, 1)
                mini_box=np.zeros((100,5), dtype=np.float32)
                mini_box_xywh=np.zeros((100,5), dtype=np.float32)
                # make sure we don't have a tiny image on the edge
                if y0+sliceHeight > image0.shape[0]:
                    y = image0.shape[0] - sliceHeight
                else:
                    y = y0
                if x0+sliceWidth > image0.shape[1]:
                    x = image0.shape[1] - sliceWidth
                else:
                    x = x0
                i_mini_box=0
                window_c = image0[y:y + sliceHeight, x:x + sliceWidth]
                for i in range(num_box):
                    if bbox[i,0]>x and bbox[i,0]<(x+sliceWidth) and bbox[i,1]>y and bbox[i,1]<(y+sliceHeight):       ##左上角坐标在框内        
                        mini_box[i_mini_box,0]=bbox[i,0]-x+1
                        mini_box[i_mini_box,1]=bbox[i,1]-y+1
                        mini_box[i_mini_box,4]=1
                        if bbox[i,2]>(x+sliceWidth): 
                            mini_box[i_mini_box,2]=sliceWidth
                        else:
                            mini_box[i_mini_box,2]=bbox[i,2]-x
                        if bbox[i,3]>(y+sliceHeight): 
                            mini_box[i_mini_box,3]=sliceHeight
                        else:
                            mini_box[i_mini_box,3]=bbox[i,3]-y   
                        if (mini_box[i_mini_box,3]-mini_box[i_mini_box,1])>30 and (mini_box[i_mini_box,2]-mini_box[i_mini_box,0])>30:
                            mini_box_xywh[i_mini_box,0]=(mini_box[i_mini_box,2]+mini_box[i_mini_box,0])/2
                            mini_box_xywh[i_mini_box,1]=(mini_box[i_mini_box,3]+mini_box[i_mini_box,1])/2
                            mini_box_xywh[i_mini_box,2]=(mini_box[i_mini_box,2]-mini_box[i_mini_box,0])
                            mini_box_xywh[i_mini_box,3]=(mini_box[i_mini_box,3]-mini_box[i_mini_box,1])
                            mini_box_xywh[i_mini_box,4]=1
#                            cv2.rectangle(window_c, (mini_box[i_mini_box,0],mini_box[i_mini_box,1]), (mini_box[i_mini_box,2],mini_box[i_mini_box,3]), (0,255,0), 2)
                            i_mini_box=i_mini_box+1
                        else:
                            mini_box[i_mini_box,:]=0
                    elif bbox[i,2]>x and bbox[i,2]<(x+sliceWidth) and bbox[i,3]>y and bbox[i,3]<(y+sliceHeight):    ##右下角坐标在框内   
                        mini_box[i_mini_box,2]=bbox[i,2]-x
                        mini_box[i_mini_box,3]=bbox[i,3]-y
                        mini_box[i_mini_box,4]=1
                        if bbox[i,0]<x: 
                            mini_box[i_mini_box,0]=0
                        else:
                            mini_box[i_mini_box,0]=bbox[i,0]-x
                        if bbox[i,1]<y: 
                            mini_box[i_mini_box,1]=0
                        else:
                            mini_box[i_mini_box,1]=bbox[i,1]-y  
                        if (mini_box[i_mini_box,3]-mini_box[i_mini_box,1])>30 and (mini_box[i_mini_box,2]-mini_box[i_mini_box,0])>30:
#                            cv2.rectangle(window_c, (mini_box[i_mini_box,0],mini_box[i_mini_box,1]), (mini_box[i_mini_box,2],mini_box[i_mini_box,3]), (0,255,0), 2)
                            
                            mini_box_xywh[i_mini_box,0]=(mini_box[i_mini_box,2]+mini_box[i_mini_box,0])/2
                            mini_box_xywh[i_mini_box,1]=(mini_box[i_mini_box,3]+mini_box[i_mini_box,1])/2
                            mini_box_xywh[i_mini_box,2]=(mini_box[i_mini_box,2]-mini_box[i_mini_box,0])
                            mini_box_xywh[i_mini_box,3]=(mini_box[i_mini_box,3]-mini_box[i_mini_box,1])
                            mini_box_xywh[i_mini_box,4]=1
                            i_mini_box=i_mini_box+1
                        else:
                            mini_box[i_mini_box,:]=0
                    elif bbox[i,0]>x and bbox[i,0]<(x+sliceWidth) and bbox[i,3]>y and bbox[i,3]<(y+sliceHeight):   ##左下角坐标在框内
                        mini_box[i_mini_box,0]=bbox[i,0]-x
                        mini_box[i_mini_box,3]=bbox[i,3]-y
                        mini_box[i_mini_box,4]=1
                        if bbox[i,2]>(x+sliceWidth): 
                            mini_box[i_mini_box,2]=sliceWidth
                        else:
                            mini_box[i_mini_box,2]=bbox[i,2]-x
                        if bbox[i,1]<y: 
                            mini_box[i_mini_box,1]=0
                        else:
                            mini_box[i_mini_box,1]=bbox[i,1]-y  
                        if (mini_box[i_mini_box,3]-mini_box[i_mini_box,1])>30 and (mini_box[i_mini_box,2]-mini_box[i_mini_box,0])>30:
#                            cv2.rectangle(window_c, (mini_box[i_mini_box,0],mini_box[i_mini_box,1]), (mini_box[i_mini_box,2],mini_box[i_mini_box,3]), (0,255,0), 2)
                            
                            mini_box_xywh[i_mini_box,0]=(mini_box[i_mini_box,2]+mini_box[i_mini_box,0])/2
                            mini_box_xywh[i_mini_box,1]=(mini_box[i_mini_box,3]+mini_box[i_mini_box,1])/2
                            mini_box_xywh[i_mini_box,2]=(mini_box[i_mini_box,2]-mini_box[i_mini_box,0])
                            mini_box_xywh[i_mini_box,3]=(mini_box[i_mini_box,3]-mini_box[i_mini_box,1])
                            mini_box_xywh[i_mini_box,4]=1
                            i_mini_box=i_mini_box+1
                        else:
                            mini_box[i_mini_box,:]=0
                    elif bbox[i,2]>x and bbox[i,2]<(x+sliceWidth) and bbox[i,1]>y and bbox[i,1]<(y+sliceHeight):   ##右上角坐标在框内   
                        mini_box[i_mini_box,2]=bbox[i,2]-x
                        mini_box[i_mini_box,1]=bbox[i,1]-y
                        mini_box[i_mini_box,4]=1
                        if bbox[i,0]<x: 
                            mini_box[i_mini_box,0]=0
                        else:
                            mini_box[i_mini_box,0]=bbox[i,0]-x
                        if bbox[i,3]>(y+sliceHeight): 
                            mini_box[i_mini_box,3]=sliceHeight
                        else:
                            mini_box[i_mini_box,3]=bbox[i,3]-y   
                        if (mini_box[i_mini_box,3]-mini_box[i_mini_box,1])>30 and (mini_box[i_mini_box,2]-mini_box[i_mini_box,0])>30:
#                            cv2.rectangle(window_c, (mini_box[i_mini_box,0],mini_box[i_mini_box,1]), (mini_box[i_mini_box,2],mini_box[i_mini_box,3]), (0,255,0), 2)
                             
                            mini_box_xywh[i_mini_box,0]=(mini_box[i_mini_box,2]+mini_box[i_mini_box,0])/2
                            mini_box_xywh[i_mini_box,1]=(mini_box[i_mini_box,3]+mini_box[i_mini_box,1])/2
                            mini_box_xywh[i_mini_box,2]=(mini_box[i_mini_box,2]-mini_box[i_mini_box,0])
                            mini_box_xywh[i_mini_box,3]=(mini_box[i_mini_box,3]-mini_box[i_mini_box,1])
                            mini_box_xywh[i_mini_box,4]=1
                            i_mini_box=i_mini_box+1 
                        else:
                            mini_box[i_mini_box,:]=0
                mini_box_xywh=np.array(mini_box_xywh, dtype=np.float32).flatten().tolist() 
#                img_raw = window_c.tobytes()
                cv2.imshow('img',window_c)
                cv2.waitKey(0) 
#                total_num=len(name_file)*(math.ceil(image0.shape[0]/dx))*(math.ceil(image0.shape[1]/dy))
#                sys.stdout.write('\r>> Converting image %d/%d' % (j, int(total_num)))
#                sys.stdout.flush()  
#                j=j+1
#                example = tf.train.Example(features=tf.train.Features(feature={
#                    'xywh':
#                            tf.train.Feature(float_list=tf.train.FloatList(value=mini_box_xywh)),
#                    'img':
#                            tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
#                    }))
#                writer.write(example.SerializeToString())
#    writer.close()
#    sys.stdout.write('\n')
#    sys.stdout.flush()