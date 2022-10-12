import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_iou(anchor_boxes, gt_box):
    """
    Input:
        anchor_boxes : ndarray -> [[c_x_s, c_y_s, anchor_w, anchor_h], ..., [c_x_s, c_y_s, anchor_w, anchor_h]].
        gt_box : ndarray -> [c_x_s, c_y_s, anchor_w, anchor_h].
    Output:
        iou : ndarray -> [iou_1, iou_2, ..., iou_m], and m is equal to the number of anchor boxes.
    """
    # compute the iou between anchor box and gt box
    # First, change [c_x_s, c_y_s, anchor_w, anchor_h] ->  [xmin, ymin, xmax, ymax]
    # anchor box :
    ab_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    ab_x1y1_x2y2[:, 0] = anchor_boxes[:, 0] - anchor_boxes[:, 2] / 2  # xmin
    ab_x1y1_x2y2[:, 1] = anchor_boxes[:, 1] - anchor_boxes[:, 3] / 2  # ymin
    ab_x1y1_x2y2[:, 2] = anchor_boxes[:, 0] + anchor_boxes[:, 2] / 2  # xmax
    ab_x1y1_x2y2[:, 3] = anchor_boxes[:, 1] + anchor_boxes[:, 3] / 2  # ymax
    w_ab, h_ab = anchor_boxes[:, 2], anchor_boxes[:, 3]
    
    # gt_box : 
    # We need to expand gt_box(ndarray) to the shape of anchor_boxes(ndarray), in order to compute IoU easily. 
    gt_box_expand = np.repeat(gt_box, len(anchor_boxes), axis=0)

    gb_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    gb_x1y1_x2y2[:, 0] = gt_box_expand[:, 0] - gt_box_expand[:, 2] / 2 # xmin
    gb_x1y1_x2y2[:, 1] = gt_box_expand[:, 1] - gt_box_expand[:, 3] / 2 # ymin
    gb_x1y1_x2y2[:, 2] = gt_box_expand[:, 0] + gt_box_expand[:, 2] / 2 # xmax
    gb_x1y1_x2y2[:, 3] = gt_box_expand[:, 1] + gt_box_expand[:, 3] / 2 # ymin
    w_gt, h_gt = gt_box_expand[:, 2], gt_box_expand[:, 3]

    # Then we compute IoU between anchor_box and gt_box
    S_gt = w_gt * h_gt
    S_ab = w_ab * h_ab
    I_w = np.minimum(gb_x1y1_x2y2[:, 2], ab_x1y1_x2y2[:, 2]) - np.maximum(gb_x1y1_x2y2[:, 0], ab_x1y1_x2y2[:, 0])
    I_h = np.minimum(gb_x1y1_x2y2[:, 3], ab_x1y1_x2y2[:, 3]) - np.maximum(gb_x1y1_x2y2[:, 1], ab_x1y1_x2y2[:, 1])
    S_I = I_h * I_w
    U = S_gt + S_ab - S_I + 1e-20
    IoU = S_I / U
    
    return IoU


def set_anchors(anchor_size):
    """
    Input:
        anchor_size : list -> [[h_1, w_1], [h_2, w_2], ..., [h_n, w_n]].
    Output:
        anchor_boxes : ndarray -> [[0, 0, anchor_w, anchor_h],
                                   [0, 0, anchor_w, anchor_h],
                                   ...
                                   [0, 0, anchor_w, anchor_h]].
    """
    anchor_number = len(anchor_size)
    anchor_boxes = np.zeros([anchor_number, 4])
    for index, size in enumerate(anchor_size): 
        anchor_w, anchor_h = size
        anchor_boxes[index] = np.array([0, 0, anchor_w, anchor_h])
    
    return anchor_boxes


def gt_creator(input_size, strides, label_lists, anchor_size, ignore_thresh):
    """制作训练正样本"""
    batch_size = len(label_lists)
    h = w = input_size
    num_scale = len(strides)
    all_anchor_size = anchor_size
    anchor_number = len(all_anchor_size) // num_scale

    gt_tensor = []
    for s in strides:
        gt_tensor.append(np.zeros([batch_size, h//s, w//s, anchor_number, 1+1+4+1+4]))
    
    # generate gt datas
    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            # get a bbox coords
            gt_class = int(gt_label[-1])
            xmin, ymin, xmax, ymax = gt_label[:-1]
            # 计算真实框的中心点和宽高
            c_x = (xmax + xmin) / 2 * w
            c_y = (ymax + ymin) / 2 * h
            box_w = (xmax - xmin) * w
            box_h = (ymax - ymin) * h

            # 检查数据
            if box_w < 1. or box_h < 1.:
                # print('A dirty data !!!')
                continue    

            # 计算先验框和边界框之间的IoU
            anchor_boxes = set_anchors(all_anchor_size)
            gt_box = np.array([[0, 0, box_w, box_h]])
            iou = compute_iou(anchor_boxes, gt_box)

            # 阈值筛选
            iou_mask = (iou > ignore_thresh)

            if iou_mask.sum() == 0:
                # 若所有的IoU都小于ignore，则将IoU最大的先验框分配给真实框，其他均视为负样本
                index = np.argmax(iou)
                # 确定该正样本被分配到哪个尺度上去，以及哪个先验框被选中为正样本
                s_indx = index // anchor_number
                ab_ind = index - s_indx * anchor_number
                # 获得该尺度的降采样倍数
                s = strides[s_indx]
                # 获得该先验框的参数
                p_w, p_h = anchor_boxes[index, 2], anchor_boxes[index, 3]
                # 计算中心点所在的网格坐标
                c_x_s = c_x / s
                c_y_s = c_y / s
                grid_x = int(c_x_s)
                grid_y = int(c_y_s)
                # 制作学习标签
                tx = c_x_s - grid_x
                ty = c_y_s - grid_y
                tw = np.log(box_w / p_w)
                th = np.log(box_h / p_h)
                weight = 2.0 - (box_w / w) * (box_h / h)

                if grid_y < gt_tensor[s_indx].shape[1] and grid_x < gt_tensor[s_indx].shape[2]:
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 0] = 1.0
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 1] = gt_class
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 2:6] = np.array([tx, ty, tw, th])
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 6] = weight
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 7:] = np.array([xmin, ymin, xmax, ymax])
            
            else:
                # 至少有一个IoU大于ignore
                
                # 我们只保留IoU最大的作为正样本，
                # 其余的要么被忽略，要么视为负样本
                best_index = np.argmax(iou)
                for index, iou_m in enumerate(iou_mask):
                    if iou_m:
                        if index == best_index:
                            # 确定该正样本被分配到哪个尺度上去，以及哪个先验框被选中为正样本
                            s_indx = index // anchor_number
                            ab_ind = index - s_indx * anchor_number
                            # 获得该尺度的降采样倍数
                            s = strides[s_indx]
                            # 获得该先验框的参数
                            p_w, p_h = anchor_boxes[index, 2], anchor_boxes[index, 3]
                            # 计算中心点所在的网格坐标
                            c_x_s = c_x / s
                            c_y_s = c_y / s
                            grid_x = int(c_x_s)
                            grid_y = int(c_y_s)
                            # 制作学习标签
                            tx = c_x_s - grid_x
                            ty = c_y_s - grid_y
                            tw = np.log(box_w / p_w)
                            th = np.log(box_h / p_h)
                            weight = 2.0 - (box_w / w) * (box_h / h)

                            if grid_y < gt_tensor[s_indx].shape[1] and grid_x < gt_tensor[s_indx].shape[2]:
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 0] = 1.0
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 1] = gt_class
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 2:6] = np.array([tx, ty, tw, th])
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 6] = weight
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 7:] = np.array([xmin, ymin, xmax, ymax])
            
                        else:
                            # 这些先验框即便IoU大于ignore，但由于不是最大的
                            # 故被忽略掉
                            s_indx = index // anchor_number
                            ab_ind = index - s_indx * anchor_number
                            s = strides[s_indx]
                            c_x_s = c_x / s
                            c_y_s = c_y / s
                            grid_x = int(c_x_s)
                            grid_y = int(c_y_s)
                            gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 0] = -1.0
                            gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 6] = -1.0

    gt_tensor = [gt.reshape(batch_size, -1, 1+1+4+1+4) for gt in gt_tensor]
    gt_tensor = np.concatenate(gt_tensor, 1)
        
    return torch.from_numpy(gt_tensor).float()

    
if __name__ == "__main__":
    gt_box = np.array([[0.0, 0.0, 10, 10]])
    anchor_boxes = np.array([[0.0, 0.0, 10, 10], 
                             [0.0, 0.0, 4, 4], 
                             [0.0, 0.0, 8, 8], 
                             [0.0, 0.0, 16, 16]
                             ])
    iou = compute_iou(anchor_boxes, gt_box)
    print(iou)