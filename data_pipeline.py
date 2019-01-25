import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
def parser(filename,augmentation,img_w,img_h):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    feats = tf.parse_single_example(serialized_example,
                                          features = {
                                                      'xywh': tf.FixedLenFeature([500], tf.float32),
                                                      'img': tf.FixedLenFeature((), tf.string)})
    coord = feats['xywh']
    dx=1/700
    dy=1/700
    coord = tf.reshape(coord, [100,5])
    coord=tf.multiply(coord,[dx,dy,dx,dy,1])

    img = tf.decode_raw(feats['img'], tf.float32)
    img = tf.reshape(img, [700, 700, 3])

    if augmentation:        
        img = tf.image.random_hue(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
        img = tf.minimum(img, 1.0)
        img = tf.maximum(img, 0.0)
    img=tf.image.resize_images(img, [img_h,img_w],method=0)
    return img, coord


def data_pipeline(file_tfrecords,batch_size,img_w,img_h,augmentation):
    min_after_dequeue = 100
    capacity = min_after_dequeue + 3 * batch_size 
    img,coord=parser(file_tfrecords,augmentation,img_w,img_h)
    image_batch, label_batch = tf.train.shuffle_batch([img, coord],   
                                                        batch_size=batch_size,   
                                                        capacity=capacity,   
                                                        min_after_dequeue=min_after_dequeue  
                                                        ) 
    return image_batch, label_batch


if __name__ == '__main__':
    file_path = 'E:/Python/tensorflow/count/trainval_2018.tfrecord'
    img_w=416
    img_h=416
    dx=img_w/700
    dy=img_h/700
    imgs, true_boxes = data_pipeline(file_path,1,img_w,img_h,False)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    imgs_, true_boxes_ = sess.run([imgs, true_boxes])
    imgs_=imgs_[0]
#    plt.imshow(imgs_)
#    plt.show
    true_boxes_=true_boxes_[0]
    num=len(true_boxes_)
    for i in range(num):
        x1=int((true_boxes_[i,0]-true_boxes_[i,2]/2))
        y1=int((true_boxes_[i,1]-true_boxes_[i,3]/2))
        x2=int((true_boxes_[i,0]+true_boxes_[i,2]/2))
        y2=int((true_boxes_[i,1]+true_boxes_[i,3]/2))
        cv2.rectangle(imgs_,(x1,y1),(x2,y2),(0,255,0),2)    
#    cv2.namedWindow("Image",cv2.WINDOW_NORMAL) 
    cv2.imshow("Image", imgs_) 
    cv2.waitKey (0)