import tensorflow as tf
import config as cfg


class yolo_head:

    def __init__(self, istraining):
        self.istraining = istraining

    def conv_layer(self, bottom, size, stride, in_channels, out_channels, use_bn, name):
        with tf.variable_scope(name):
            conv = tf.layers.conv2d(bottom, out_channels, size, stride, padding="SAME",
                                    use_bias=not use_bn, activation=None)
            if use_bn:
                conv_bn = tf.layers.batch_normalization(conv, training=self.istraining)
                act = tf.nn.leaky_relu(conv_bn, 0.1)
            else:
                act = conv
        return act
    def build(self, feat_ex, res18, res10):
        self.conv52 = self.conv_layer(feat_ex, 1, 1, 1024, 512, True, 'conv_head_52')  		# 13x512
        self.conv53 = self.conv_layer(self.conv52, 3, 1, 512, 1024, True, 'conv_head_53')   # 13x1024
        self.conv54 = self.conv_layer(self.conv53, 1, 1, 1024, 512, True, 'conv_head_54')   # 13x512
        self.conv55 = self.conv_layer(self.conv54, 3, 1, 512, 1024, True, 'conv_head_55')   # 13x1024
        self.conv56 = self.conv_layer(self.conv55, 1, 1, 1024, 512, True, 'conv_head_56')   # 13x512
        self.conv57 = self.conv_layer(self.conv56, 3, 1, 512, 1024, True, 'conv_head_57')   # 13x1024
        self.conv58 = self.conv_layer(self.conv57, 1, 1, 1024, 15, False, 'conv_head_58')   # 13x15
        # follow yolo layer mask = 6,7,8
        self.conv59 = self.conv_layer(self.conv56, 1, 1, 512, 256, True, 'conv_head_59')    # 13x256
        size = tf.shape(self.conv59)[1]
        self.upsample0 = tf.image.resize_nearest_neighbor(self.conv59, [2*size, 2*size],
                                                          name='upsample_0')                # 26x256
        self.route0 = tf.concat([self.upsample0, res18], axis=-1, name='route_0')           # 26x768
        self.conv60 = self.conv_layer(self.route0, 1, 1, 768, 256, True, 'conv_head_60')    # 26x256
        self.conv61 = self.conv_layer(self.conv60, 3, 1, 256, 512, True, 'conv_head_61')    # 26x512
        self.conv62 = self.conv_layer(self.conv61, 1, 1, 512, 256, True, 'conv_head_62')    # 26x256
        self.conv63 = self.conv_layer(self.conv62, 3, 1, 256, 512, True, 'conv_head_63')    # 26x512
        self.conv64 = self.conv_layer(self.conv63, 1, 1, 512, 256, True, 'conv_head_64')    # 26x256
        self.conv65 = self.conv_layer(self.conv64, 3, 1, 256, 512, True, 'conv_head_65')    # 26x512
        self.conv66 = self.conv_layer(self.conv65, 1, 1, 512, 15, False, 'conv_head_66')    # 26x15
        # follow yolo layer mask = 3,4,5
        self.conv67 = self.conv_layer(self.conv64, 1, 1, 256, 128, True, 'conv_head_67')    # 26x128
        size = tf.shape(self.conv67)[1]
        self.upsample1 = tf.image.resize_nearest_neighbor(self.conv67, [2 * size, 2 * size],
                                                          name='upsample_1')                # 52x128
        self.route1 = tf.concat([self.upsample1, res10], axis=-1, name='route_1')           # 52x384
        self.conv68 = self.conv_layer(self.route1, 1, 1, 384, 128, True, 'conv_head_68')    # 52x128
        self.conv69 = self.conv_layer(self.conv68, 3, 1, 128, 256, True, 'conv_head_69')    # 52x256
        self.conv70 = self.conv_layer(self.conv69, 1, 1, 256, 128, True, 'conv_head_70')    # 52x128
        self.conv71 = self.conv_layer(self.conv70, 3, 1, 128, 256, True, 'conv_head_71')    # 52x256
        self.conv72 = self.conv_layer(self.conv71, 1, 1, 256, 128, True, 'conv_head_72')    # 52x128
        self.conv73 = self.conv_layer(self.conv72, 3, 1, 128, 256, True, 'conv_head_73')    # 52x256
        self.conv74 = self.conv_layer(self.conv73, 1, 1, 256, 15, False, 'conv_head_74')    # 52x15
        # follow yolo layer mask = 0,1,2

        return self.conv74, self.conv66, self.conv58

class yolo_det:
    """Convert final layer features to bounding box parameters.

        Parameters
        ----------
        feats : tensor
            Final convolutional layer features.
        anchors : array-like
            Anchor box widths and heights.
        num_classes : int
            Number of target classes.

        Returns
        -------
        box_xy : tensor
            x, y box predictions adjusted by spatial location in conv layer.
        box_wh : tensor
            w, h box predictions adjusted by anchors and conv spatial resolution.
        box_conf : tensor
            Probability estimate for whether each box contains any object.
        box_class_pred : tensor
            Probability distribution estimate for each box over class labels.
    """
    def __init__(self, anchors, img_shape):
        self.anchors = anchors
        self.img_shape = img_shape

    def build(self, feats):
        # Reshapce to bach, height, widht, num_anchors, box_params
        anchors_tensor = tf.reshape(self.anchors, [1, 1, 1, cfg.num_anchors_per_layer, 2])

        # Dynamic implementation of conv dims for fully convolutional model
        conv_dims = tf.stack([tf.shape(feats)[2], tf.shape(feats)[1]])    # assuming channels last, w h
        # In YOLO the height index is the inner most iteration
        conv_height_index = tf.range(conv_dims[1])
        conv_width_index = tf.range(conv_dims[0])
        conv_width_index, conv_height_index = tf.meshgrid(conv_width_index, conv_height_index)##输出0，0.。。0；1,1....1；
        conv_height_index = tf.reshape(conv_height_index, [-1, 1])
        conv_width_index = tf.reshape(conv_width_index, [-1, 1])
        conv_index = tf.concat([conv_width_index, conv_height_index], axis=-1)
        # 0, 0
        # 1, 0
        # 2, 0
        # ...
        # 12, 0
        # 0, 1
        # 1, 1
        # ...
        # 12, 1
        conv_index = tf.reshape(conv_index, [1, conv_dims[1], conv_dims[0], 1, 2])  # [1, 13, 13, 1, 2]
        conv_index = tf.cast(conv_index, tf.float32)

        feats = tf.reshape(
            feats, [-1, conv_dims[1], conv_dims[0], cfg.num_anchors_per_layer, 5])
        # [None, 13, 13, 3, 5]

        conv_dims = tf.cast(tf.reshape(conv_dims, [1, 1, 1, 1, 2]), tf.float32)

        img_dims = tf.stack([self.img_shape[2], self.img_shape[1]])   # w, h
        img_dims = tf.cast(tf.reshape(img_dims, [1, 1, 1, 1, 2]), tf.float32)

        box_xy = tf.sigmoid(feats[..., :2])  # σ(tx), σ(ty)     # [None, 13, 13, 3, 2] 
        box_twh = feats[..., 2:4] ## w,h
        box_wh = tf.exp(box_twh)  # exp(tw), exp(th)    # [None, 13, 13, 3, 2]
        self.box_confidence = tf.sigmoid(feats[..., 4:5])

        self.box_xy = (box_xy + conv_index) / conv_dims  # relative the whole img [0, 1]
        self.box_wh = box_wh * anchors_tensor / img_dims  # relative the whole img [0, 1]
        self.loc_txywh = tf.concat([box_xy, box_twh], axis=-1)

        return self.box_xy, self.box_wh, self.box_confidence, self.loc_txywh
        # box_xy: [None, 13, 13, 3, 2]
        # box_wh: [None, 13, 13, 3, 2]
        # box_confidence: [None, 13, 13, 3, 1]


def preprocess_true_boxes(true_boxes, anchors, feat_size, image_size): 
    """

    :param true_boxes: x, y, w, h, class  [batch,30,5]
    :param anchors: [3,2]
    :param feat_size:[batch,13,13,5]
    :param image_size:[batch,416,416,3]
    :return:
    """
    num_anchors = cfg.num_anchors_per_layer ##3

    true_wh = tf.expand_dims(true_boxes[..., 2:4], 2)  #true_boxes[..., 2:4]=[batch,30,2], [batch, 30, 1, 2]
    true_wh_half = true_wh / 2.
    true_mins = 0 - true_wh_half
    true_maxes = true_wh_half ##尺寸：[batch, 100, 1, 2],值：[-wh/2,wh/2]

    img_wh = tf.reshape(tf.stack([image_size[2], image_size[1]]), [1, -1])
    anchors = anchors / tf.cast(img_wh, tf.float32)  # normalize
    anchors_shape = tf.shape(anchors)  # [num_anchors, 2]
    anchors = tf.reshape(anchors, [1, 1, anchors_shape[0], anchors_shape[1]])  # [1, 1, num_anchors, 2]
    anchors_half = anchors / 2.
    anchors_mins = 0 - anchors_half
    anchors_maxes = anchors_half

    intersect_mins = tf.maximum(true_mins, anchors_mins) ##tf.maximum不同于tf.argmax
    intersect_maxes = tf.minimum(true_maxes, anchors_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)     # [batch, 100, num_anchors, 2]
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]   # [batch, 100, num_anchors] 每个anchor与原图img的相交面积

    true_areas = true_wh[..., 0] * true_wh[..., 1]      # [batch, 100, 1]  ##每个box的面积
    anchors_areas = anchors[..., 0] * anchors[..., 1]   # [1, 1, num_anchors] ##每个anchor的面积

    union_areas = true_areas + anchors_areas - intersect_areas  # [batch, 100, num_anchors]

    iou_scores = intersect_areas / union_areas  # [batch, 100, num_anchors] ##计算每个box与每个anchor的IOU
    ##tf.equal是判断是否相等，相等则输出true，tf.reduce_all计算一个张量纬度上的逻辑和，axis=-1表示减少一个纬度，如不加axis，则只得到1个值
    ##tf.logical_not逻辑结果取反，入原为true，运行后为false
    ##最终结果统计iou不为0，3个anchor只要有1个不为0，则结果为true
    valid = tf.logical_not(tf.reduce_all(tf.equal(iou_scores, 0), axis=-1))     # [batch, 100] 
    ##tf.argmax是取最大的下标，axis=-1是每行对比
    ##选取3个anchor中iou最大的
    iout_argmax = tf.cast(tf.argmax(iou_scores, axis=-1), tf.int32)   # [batch, 100], (0, 1, 2)
    anchors = tf.reshape(anchors, [-1, 2])      # has been normalize by img shape [num_anchor,2]
    ##最大IOU所对应的anchor大小    
    anchors_cf = tf.gather(anchors, iout_argmax)   # [batch, 100, 2]

    feat_wh = tf.reshape(tf.stack([feat_size[2], feat_size[1]]), [1, -1])  # (1, 2)
    ##得出中心点x,y在第几个bbox中，tf.floor作用是向下去整
    cxy = tf.cast(tf.floor(true_boxes[..., :2] * tf.cast(feat_wh, tf.float32)),
                  tf.int32)    # [batch, 100, 2]   bx = cx + σ(tx)
    ##得到的是相对于每个bbox，中心点的相对偏移；如原坐标点是(6.2,5.2)则运行后是(0.2,0.2)
    sig_xy = tf.cast(true_boxes[..., :2] * tf.cast(feat_wh, tf.float32) - tf.cast(cxy, tf.float32),
                     tf.float32)   # [batch, 100, 2]
    ##cxy[...,1]是代表第几行，原图中(ymax-ymin)/2得到的
    idx = cxy[..., 1] * (num_anchors * feat_size[2]) + num_anchors * cxy[..., 0] + iout_argmax  # [batch, 30]
    idx_one_hot = tf.one_hot(idx, depth=feat_size[1] * feat_size[2] * num_anchors)   # [batch, 100, 13x13x3]
    idx_one_hot = tf.reshape(idx_one_hot,
                        [-1, cfg.max_truth, feat_size[1], feat_size[2], num_anchors,
                         1])  # (batch, 100, 13, 13, 3, 1)
    loc_scale = 2 - true_boxes[..., 2] * true_boxes[..., 3]     # ???意义是什么？？(batch, 100)
    mask = []
    loc_cls = [] 
    scale = []
    for i in range(cfg.batch_size):
        idx_i = tf.where(valid[i])[:, 0]    # (?, )    # false / true
        mask_i = tf.gather(idx_one_hot[i], idx_i)   # (?, 13, 13, 3, 1)
        a=idx_one_hot[i]
        mask_i1111 = tf.gather(idx_one_hot[i], idx_i) 
        scale_i = tf.gather(loc_scale[i], idx_i)    # (?, )
        scale_i = tf.reshape(scale_i, [-1, 1, 1, 1, 1])     # (?, 1, 1, 1, 1)
        scale_i = scale_i * mask_i      # (?, 13, 13, 3, 1)
        scale_i = tf.reduce_sum(scale_i, axis=0)        # (13, 13, 3, 1)
        scale_i = tf.maximum(tf.minimum(scale_i, 2), 1)
        scale.append(scale_i) ##最终得到的和one_hot形式差不多，但1的值改成了loc_scale，有多少个idx_i，纬度(13,13,3,1)就有多少个loc_scale

        true_boxes_i = tf.gather(true_boxes[i], idx_i)    # (?, 5)
        sig_xy_i = tf.gather(sig_xy[i], idx_i)    # (?, 2)
        anchors_cf_i = tf.gather(anchors_cf[i], idx_i)    # (?, 2)
        twh = tf.log(true_boxes_i[:, 2:4] / anchors_cf_i)
        loc_cls_i = tf.concat([sig_xy_i, twh, true_boxes_i[:, -1:]], axis=-1)    # (?, 5)
        loc_cls_i = tf.reshape(loc_cls_i, [-1, 1, 1, 1, 5])     # (?, 1, 1, 1, 5)
        loc_cls_i = loc_cls_i * mask_i      # (?, 13, 13, 3, 5)
        loc_cls_i = tf.reduce_sum(loc_cls_i, axis=[0])  # (13, 13, 3, 5)
        # exception, one anchor is responsible for 2 or more object
        loc_cls_i = tf.concat([loc_cls_i[..., :4], loc_cls_i[..., -1:]], axis=-1)
        loc_cls.append(loc_cls_i)

        mask_i = tf.reduce_sum(mask_i, axis=[0])    # (13, 13, 3, 1)
        mask_i = tf.minimum(mask_i, 1)
        mask.append(mask_i)
    loc_cls = tf.stack(loc_cls, axis=0)     # (σ(tx), σ(tx), tw, th, cls)
    mask = tf.stack(mask, axis=0)
    scale = tf.stack(scale, axis=0)
    return loc_cls, mask, scale,a,idx,idx_i

def box_IoU(b1, b2):
    """
    Calculer IoU between 2 BBs
    # hoi bi nguoc han tinh left bottom, right top TODO
    :param b1: predicted box, shape=[None, 13, 13, 3, 4], 4: xywh
    :param b2: true box, shape=[None, 13, 13, 3, 4], 4: xywh
    :return: iou: intersection of 2 BBs, tensor, shape=[None, 13, 13, 3, 1] ,1: IoU
    b = tf.cast(b, dtype=tf.float32)
    """
    with tf.name_scope('BB1'):
        """Calculate 2 corners: {left bottom, right top} and area of this box"""
        b1 = tf.expand_dims(b1, -2)  # shape= (None, 13, 13, 3, 1, 4)
        b1_xy = b1[..., :2]  # x,y shape=(None, 13, 13, 3, 1, 2)
        b1_wh = b1[..., 2:4]  # w,h shape=(None, 13, 13, 3, 1, 2)
        b1_wh_half = b1_wh / 2.  # w/2, h/2 shape= (None, 13, 13, 3, 1, 2)
        b1_mins = b1_xy - b1_wh_half  # x,y: left bottom corner of BB
        b1_maxes = b1_xy + b1_wh_half  # x,y: right top corner of BB
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]  # w1 * h1 (None, 13, 13, 3, 1)

    with tf.name_scope('BB2'):
        """Calculate 2 corners: {left bottom, right top} and area of this box"""
        b2 = tf.expand_dims(b2, -2)  # shape= (None, 13, 13, 3, 1, 4)
        #b2 = tf.expand_dims(b2, 0)  # shape= (1, None, 13, 13, 3, 4)  # TODO 0?
        b2_xy = b2[..., :2]  # x,y shape=(None, 13, 13, 3, 1, 2)
        b2_wh = b2[..., 2:4]  # w,h shape=(None, 13, 13, 3, 1, 2)
        b2_wh_half = b2_wh / 2.  # w/2, h/2 shape=(None, 13, 13, 3, 1, 2)
        b2_mins = b2_xy - b2_wh_half  # x,y: left bottom corner of BB
        b2_maxes = b2_xy + b2_wh_half  # x,y: right top corner of BB
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]  # w2 * h2

    with tf.name_scope('Intersection'):
        """Calculate 2 corners: {left bottom, right top} based on BB1, BB2 and area of this box"""
        # intersect_mins = tf.maximum(b1_mins, b2_mins, name='left_bottom')  # (None, 13, 13, 3, 1, 2)
        intersect_mins = tf.maximum(b1_mins, b2_mins)  # (None, 13, 13, 3, 1, 2)
        # intersect_maxes = tf.minimum(b1_maxes, b2_maxes, name='right_top')  #
        intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
        # intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)  # (None, 13, 13, 3, 1, 2), 2: w,h
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]  # intersection: wi * hi (None, 13, 13, 3, 1)

    IoU = tf.divide(intersect_area, (b1_area + b2_area - intersect_area), name='divise-IoU')  # (None, 13, 13, 3, 1)

    return IoU
def confidence_loss(pred_xy, pred_wh, pred_confidence, true_boxes, detectors_mask):
    """

    :param pred_xy: [batch, 13, 13, 3, 2] from yolo_det
    :param pred_wh: [batch, 13, 13, 3, 2] from yolo_det
    :param pred_confidence: [batch, 13, 13, 3, 1] from yolo_det
    :param true_boxes: [batch, 100, 5]
    :param detectors_mask: [batch, 13, 13, 3, 1]
    :return:
    """
    pred_xy = tf.expand_dims(pred_xy, 4)  # [batch, 13, 13, 3, 1, 2]
    pred_wh = tf.expand_dims(pred_wh, 4)  # [batch, 13, 13, 3, 1, 2]

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    true_boxes_shape = tf.shape(true_boxes)  # [batch, num_true_boxes, box_params(5)]
    true_boxes = tf.reshape(true_boxes, [
        true_boxes_shape[0], 1, 1, 1, true_boxes_shape[1], true_boxes_shape[2]
        ])  # [batch, 1, 1, 1, num_true_boxes, 5]
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]

    # Find IOU of each predicted box with each ground truth box.
    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    # [batch, 13, 13, 3, 1, 2] [batch, 1, 1, 1, num_true_boxes, 2]
    intersect_mins = tf.maximum(pred_mins, true_mins)
    # [batch, 13, 13, 3, num_true_boxes, 2]
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    # [batch, 13, 13, 3, num_true_boxes, 2]
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    # [batch, 13, 13, 3, num_true_boxes, 2]
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    # [batch, 13, 13, 3, num_true_boxes]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    # [batch, 13, 13, 3, 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]
    # [batch, 1, 1, 1, num_true_boxes]
    union_areas = pred_areas + true_areas - intersect_areas
    # [batch, 13, 13, 3, num_true_boxes]
    iou_scores = intersect_areas / union_areas
    best_ious = tf.reduce_max(iou_scores, axis=-1, keepdims=True)  # Best IOU scores.
    ignore_mask = tf.cast(best_ious < cfg.ignore_thresh, best_ious.dtype)##大于阈值的框为1
    no_objects_loss = (1- detectors_mask)*ignore_mask*tf.square(pred_confidence)
   # no_objects_loss = no_object_weights * tf.square(pred_confidence)##算没有物体的loss，如果预测与标签置信度一致，则此致刚好为0
    objects_loss = detectors_mask *tf.square(1-pred_confidence)##同理上面
    objectness_loss = tf.reduce_sum(objects_loss)/cfg.batch_size
    no_objects_loss_mean=tf.reduce_sum(no_objects_loss)/cfg.batch_size
    return objectness_loss,no_objects_loss_mean


def cord_cls_loss(
                detectors_mask,
                matching_true_boxes,
                pred_boxes,
                loc_scale,
              ):
    """
    :param detectors_mask: [batch, 13, 13, 3, 1]
    :param matching_true_boxes: [batch, 13, 13, 3, 5]   [σ(tx), σ(ty), tw, th, cls]
    :param pred_boxes: [batch, 13, 13, 3, 4]
    :param loc_scale: [batch, 13, 13, 3, 1]
    :return:
        mean_loss: float
        mean localization loss across minibatch
    """

    # Classification loss for matching detections.
    # NOTE: YOLO does not use categorical cross-entropy loss here.                             
    # Coordinate loss for matching detection boxes.   [σ(tx), σ(ty), tw, th]
#    matching_boxes_xy = matching_true_boxes[..., 0:2]
#    matching_boxes_wh = matching_true_boxes[..., 2:4]
    matching_boxes = matching_true_boxes[..., 0:4]
#    pred_boxes_xy=pred_boxes[...,0:2]
#    pred_boxes_wh=pred_boxes[...,2:4]
    iou_scores=box_IoU(matching_boxes,pred_boxes)
    iou_sum=tf.reduce_sum(iou_scores*detectors_mask)
    num_obj=tf.reduce_sum(detectors_mask)
    avg_iou=iou_sum/num_obj
    coordinates_loss = detectors_mask * loc_scale * tf.square(matching_boxes - pred_boxes)
#    coordinates_loss_xy = detectors_mask * loc_scale*tf.nn.softmax_cross_entropy_with_logits_v2(labels = matching_boxes_xy, logits = pred_boxes_xy)
    coordinates_loss_sum = tf.reduce_sum(coordinates_loss)/cfg.batch_size
    
    

    return coordinates_loss_sum,avg_iou
if __name__ == "__main__":
    true_boxes=tf.placeholder(tf.float32, shape=[32,30,5] )
    anchors=tf.placeholder(tf.float32, shape=[3,2] )
    feat_size=tf.placeholder(tf.float32, shape=[32,13,13,75] )
    feat_size=tf.shape(feat_size)
    image_size=tf.placeholder(tf.float32, shape=[32,416,416,3] )
    image_size=tf.shape(image_size)
    pred_xy=tf.placeholder(tf.float32, shape=[32,13,13,3,2] )
    pred_wh=tf.placeholder(tf.float32, shape=[32,13,13,3,2] )
    pred_confidence=tf.placeholder(tf.float32, shape=[32,13,13,3,1] )
    detectors_mask=tf.placeholder(tf.float32, shape=[32,13,13,3,1] )
    a=confidence_loss(pred_xy, pred_wh, pred_confidence, true_boxes, detectors_mask)