import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import pandas as pd
import cv2
#def convert_coco_bbox(size, box):
#    """
#    Introduction
#    ------------
#        计算box的长宽和原始图像的长宽比值
#    Parameters
#    ----------
#        size: 原始图像大小
#        box: 标注box的信息
#    Returns
#        x, y, w, h 标注box和原始图像的比值
#    """
#    dw = 1. / size[0]
#    dh = 1. / size[1]
#    x = (box[0] + box[2]) / 2.0 - 1
#    y = (box[1] + box[3]) / 2.0 - 1
#    w = box[2]
#    h = box[3]
#    x = x * dw
#    w = w * dw
#    y = y * dh
#    h = h * dh
#    return x, y, w, h

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0-1
    y = (box[2] + box[3])/2.0-1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [x, y, w, h]
def box_iou(boxes, clusters):
    """
    Introduction
    ------------
        计算每个box和聚类中心的距离值
    Parameters
    ----------
        boxes: 所有的box数据
        clusters: 聚类中心
    """
    box_num = boxes.shape[0]
    cluster_num = clusters.shape[0]
    box_area = boxes[:, 0] * boxes[:, 1]
    #每个box的面积重复9次，对应9个聚类中心
    box_area = box_area.repeat(cluster_num)
    box_area = np.reshape(box_area, [box_num, cluster_num])

    cluster_area = clusters[:, 0] * clusters[:, 1]
    cluster_area = np.tile(cluster_area, [1, box_num])
    cluster_area = np.reshape(cluster_area, [box_num, cluster_num])

    #这里计算两个矩形的iou，默认所有矩形的左上角坐标都是在原点，然后计算iou，因此只需取长宽最小值相乘就是重叠区域的面积
    boxes_width = np.reshape(boxes[:, 0].repeat(cluster_num), [box_num, cluster_num])
    clusters_width = np.reshape(np.tile(clusters[:, 0], [1, box_num]), [box_num, cluster_num])
    min_width = np.minimum(clusters_width, boxes_width)

    boxes_high = np.reshape(boxes[:, 1].repeat(cluster_num), [box_num, cluster_num])
    clusters_high = np.reshape(np.tile(clusters[:, 1], [1, box_num]), [box_num, cluster_num])
    min_high = np.minimum(clusters_high, boxes_high)

    iou = np.multiply(min_high, min_width) / (box_area + cluster_area - np.multiply(min_high, min_width))
    return iou


def avg_iou(boxes, clusters):
    """
    Introduction
    ------------
        计算所有box和聚类中心的最大iou均值作为准确率
    Parameters
    ----------
        boxes: 所有的box
        clusters: 聚类中心
    Returns
    -------
        accuracy: 准确率
    """
    return np.mean(np.max(box_iou(boxes, clusters), axis =1))


def Kmeans(boxes, cluster_num, iteration_cutoff = 25, function = np.median):
    """
    Introduction
    ------------
        根据所有box的长宽进行Kmeans聚类
    Parameters
    ----------
        boxes: 所有的box的长宽
        cluster_num: 聚类的数量
        iteration_cutoff: 当准确率不再降低多少轮停止迭代
        function: 聚类中心更新的方式
    Returns
    -------
        clusters: 聚类中心box的大小
    """
    boxes_num = boxes.shape[0]
    best_average_iou = 0
    best_avg_iou_iteration = 0
    best_clusters = []
    anchors = []
    np.random.seed()
    # 随机选择所有boxes中的box作为聚类中心
    clusters = boxes[np.random.choice(boxes_num, cluster_num, replace = False)]
    count = 0
    while True:
        distances = 1. - box_iou(boxes, clusters)
        boxes_iou = np.min(distances, axis=1)
        # 获取每个box距离哪个聚类中心最近
        current_box_cluster = np.argmin(distances, axis=1)
        average_iou = np.mean(1. - boxes_iou)
        if average_iou > best_average_iou:
            best_average_iou = average_iou
            best_clusters = clusters
            best_avg_iou_iteration = count
        # 通过function的方式更新聚类中心
        for cluster in range(cluster_num):
            clusters[cluster] = function(boxes[current_box_cluster == cluster], axis=0)
        if count >= best_avg_iou_iteration + iteration_cutoff:
            break
        print("Sum of all distances (cost) = {}".format(np.sum(boxes_iou)))
        print("iter: {} Accuracy: {:.2f}%".format(count, avg_iou(boxes, clusters) * 100))
        count += 1
    for cluster in best_clusters:
        anchors.append([round(cluster[0]), round(cluster[1])])
    return anchors, best_average_iou




def convert_annotation(image_id):
    in_file = open('E:/Python/tensorflow/YOLO/pascal_VOC/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations/%s.xml'%(image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for i, obj in enumerate(root.iter('object')):
        if i > 29:
            break
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
    return bb
def process(cluster_num, iteration_cutoff = 25, function = np.median):
    """
    Introduction
    ------------
        主处理函数
    Parameters
    ----------
        dataFile: 数据集的标注文件
        cluster_num: 聚类中心数目
        iteration_cutoff: 当准确率不再降低多少轮停止迭代
        function: 聚类中心更新的方式
    """
#    image_ids = open('E:/Python/tensorflow/YOLO/pascal_VOC/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt' ).read().strip().split()
    # print(filename)
#    boxes=[]
#    for image_id in image_ids:
#        bb = convert_annotation(image_id)
#        boxes.append(bb[2:])
#    boxes=np.array(boxes)
    boxes=make_xywh()
    last_best_iou = 0
    last_anchors = []
    box_w = boxes[:10000, 0]
    box_h = boxes[:10000, 1]
    plt.scatter(box_h, box_w, c = 'r')
    anchors = Kmeans(boxes, cluster_num, iteration_cutoff, function)
    anchors=np.array(anchors[0])
    plt.scatter(anchors[:,0], anchors[:, 1], c = 'b')
    plt.show()
    for _ in range(100):
        anchors, best_iou = Kmeans(boxes, cluster_num, iteration_cutoff, function)
        if best_iou > last_best_iou:
            last_anchors = anchors
            last_best_iou = best_iou
            print("anchors: {}, avg iou: {}".format(last_anchors, last_best_iou))
    print("final anchors: {}, avg iou: {}".format(last_anchors, last_best_iou))

def make_xywh():
    max_box=500
    sliceHeight=700 
    sliceWidth=700
    overlap=0.2  
    data = pd.read_table("E:/Python/tensorflow/count/train_labels.csv",sep=",")
    name_file=open('E:/Python/tensorflow/count/train.txt','r')
    name_file=name_file.readlines()
    box=[]
    for name in name_file:
        name=name[:-1]
        image_path='E:/Python/tensorflow/count/train_dataset/'+name
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
                            box.append(mini_box_xywh[i_mini_box,2:4])
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
                            box.append(mini_box_xywh[i_mini_box,2:4])
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
                            box.append(mini_box_xywh[i_mini_box,2:4])
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
                            box.append(mini_box_xywh[i_mini_box,2:4])
                            i_mini_box=i_mini_box+1 
                        else:
                            mini_box[i_mini_box,:]=0
                mini_box_xywh=np.array(mini_box_xywh, dtype=np.float32).flatten().tolist() 
    boxes=np.array(box)
    return  boxes
if __name__ == '__main__':
    process(9)
