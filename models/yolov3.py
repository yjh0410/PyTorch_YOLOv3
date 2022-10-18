import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import Conv, SPP
from backbone import build_backbone

import numpy as np
from .loss import iou_score, compute_loss


class YOLOv3(nn.Module):
    def __init__(self,
                 cfg,
                 device,
                 input_size=None,
                 num_classes=20,
                 trainable=False,
                 conf_thresh=0.001, 
                 nms_thresh=0.50, 
                 topk=100,
                 anchor_size=None):
        super(YOLOv3, self).__init__()
        self.cfg = cfg
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        self.stride = cfg['stride']

        # Anchor box config
        self.anchor_size = torch.tensor(anchor_size).view(len(cfg['stride']), len(anchor_size) // 3, 2) # [S, KA, 2]
        self.num_anchors = self.anchor_size.size(1)
        self.anchor_boxes = self.create_grid(input_size)

        # 主干网络
        self.backbone, feat_dims = build_backbone(cfg['backbone'], cfg['pretrained'])
        
        # s = 8
        self.conv_set_1 = nn.Sequential(
            Conv(feat_dims[-3]+feat_dims[-2]//4, feat_dims[-3]//2, k=1),
            Conv(feat_dims[-3]//2, feat_dims[-3], k=3, p=1),
            Conv(feat_dims[-3], feat_dims[-3]//2, k=1),
            Conv(feat_dims[-3]//2, feat_dims[-3], k=3, p=1),
            Conv(feat_dims[-3], feat_dims[-3]//2, k=1)
        )
        self.extra_conv_1 = Conv(feat_dims[-3]//2, cfg['head_dim'][-3], k=3, p=1)
        self.pred_1 = nn.Conv2d(cfg['head_dim'][-3], self.num_anchors*(1 + 4 + self.num_classes), kernel_size=1)
    
        # s = 16
        self.conv_set_2 = nn.Sequential(
            Conv(feat_dims[-2]+feat_dims[-1]//4, feat_dims[-2]//2, k=1),
            Conv(feat_dims[-2]//2, feat_dims[-2], k=3, p=1),
            Conv(feat_dims[-2], feat_dims[-2]//2, k=1),
            Conv(feat_dims[-2]//2, feat_dims[-2], k=3, p=1),
            Conv(feat_dims[-2], feat_dims[-2]//2, k=1)
        )
        self.conv_1x1_2 = Conv(feat_dims[-2]//2, feat_dims[-2]//4, k=1)
        self.extra_conv_2 = Conv(feat_dims[-2]//2, cfg['head_dim'][-2], k=3, p=1)
        self.pred_2 = nn.Conv2d(cfg['head_dim'][-2], self.num_anchors*(1 + 4 + self.num_classes), kernel_size=1)

        # s = 32
        self.conv_set_3 = nn.Sequential(
            Conv(feat_dims[-1], feat_dims[-1]//2, k=1),
            Conv(feat_dims[-1]//2, feat_dims[-1], k=3, p=1),
            Conv(feat_dims[-1], feat_dims[-1]//2, k=1),
            Conv(feat_dims[-1]//2, feat_dims[-1], k=3, p=1),
            Conv(feat_dims[-1], feat_dims[-1]//2, k=1)
        )
        self.conv_1x1_3 = Conv(feat_dims[-1]//2, feat_dims[-1]//4, k=1)
        self.extra_conv_3 = Conv(feat_dims[-1]//2, cfg['head_dim'][-1], k=3, p=1)
        self.pred_3 = nn.Conv2d(cfg['head_dim'][-1], self.num_anchors*(1 + 4 + self.num_classes), kernel_size=1)


        if self.trainable:
            self.init_bias()


    def init_bias(self):
        # init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        for pred in [self.pred_1, self.pred_2, self.pred_3]:
            nn.init.constant_(pred.bias[..., :self.num_anchors], bias_value)
            nn.init.constant_(pred.bias[..., 1*self.num_anchors:(1+self.num_classes)*self.num_anchors], bias_value)


    def create_grid(self, input_size):
        all_anchor_boxes = []

        for level, stride in enumerate(self.stride):
            # generate grid cells
            fmp_w, fmp_h = input_size // stride, input_size // stride
            grid_y, grid_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
            # [H, W, 2] -> [HW, 2]
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)
            # [HW, 2] -> [HW, 1, 2] -> [HW, KA, 2]
            grid_xy = grid_xy[:, None, :].repeat(1, self.num_anchors, 1)

            # [KA, 2]
            anchor_size = self.anchor_size[level]
            # [KA, 2] -> [1, KA, 2] -> [HW, KA, 2]
            anchor_wh = anchor_size[None, :, :].repeat(fmp_h*fmp_w, 1, 1)

            # [HW, KA, 4] -> [M, 4]
            anchor_boxes = torch.cat([grid_xy, anchor_wh], dim=-1)
            anchor_boxes = anchor_boxes.view(-1, 4).to(self.device)

            all_anchor_boxes.append(anchor_boxes)

        return all_anchor_boxes


    def set_grid(self, input_size):
        self.input_size = input_size
        self.anchor_boxes = self.create_grid(input_size)


    def decode_boxes(self, anchors, txtytwth_pred, stride):
        """将txtytwth预测换算成边界框的左上角点坐标和右下角点坐标 \n
            Input: \n
                txtytwth_pred : [B, H*W*KA, 4] \n
            Output: \n
                x1y1x2y2_pred : [B, H*W*KA, 4] \n
        """
        # 获得边界框的中心点坐标和宽高
        # b_x = sigmoid(tx) + gride_x
        # b_y = sigmoid(ty) + gride_y
        xy_pred = (torch.sigmoid(txtytwth_pred[..., :2]) + anchors[..., :2]) * stride
        # b_w = anchor_w * exp(tw)
        # b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[..., 2:]) * anchors[..., 2:]

        # [B, H*W*KA, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1)

        # 将中心点坐标和宽高换算成边界框的左上角点坐标和右下角点坐标
        x1y1x2y2_pred = torch.zeros_like(xywh_pred)
        x1y1x2y2_pred[..., :2] = xywh_pred[..., :2] - xywh_pred[..., 2:] * 0.5
        x1y1x2y2_pred[..., 2:] = xywh_pred[..., :2] + xywh_pred[..., 2:] * 0.5
        
        return x1y1x2y2_pred


    def nms(self, bboxes, scores):
        """"Pure Python NMS baseline."""
        x1 = bboxes[:, 0]  #xmin
        y1 = bboxes[:, 1]  #ymin
        x2 = bboxes[:, 2]  #xmax
        y2 = bboxes[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []                                             
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # 计算交集的左上角点和右下角点的坐标
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # 计算交集的宽高
            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            # 计算交集的面积
            inter = w * h

            # 计算交并比
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            # 滤除超过nms阈值的检测框
            inds = np.where(iou <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def postprocess(self, conf_preds, cls_preds, reg_preds):
        """
        Input:
            conf_preds: List(Tensor) [[H*W*KA, 1], ...]
            cls_preds:  List(Tensor) [[H*W*KA, C], ...]
            reg_preds:  List(Tensor) [[H*W*KA, 4], ...]
        """

        all_scores = []
        all_labels = []
        all_bboxes = []
        anchors = self.anchor_boxes

        for level, (conf_pred_i, cls_pred_i, reg_pred_i, anchors_i) \
                in enumerate(zip(conf_preds, cls_preds, reg_preds, anchors)):
            # (H x W x KA x C,)
            scores_i = torch.sigmoid(conf_pred_i) * torch.softmax(cls_pred_i, dim=-1).flatten()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk, reg_pred_i.size(0))

            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = scores_i.sort(descending=True)
            topk_scores = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = topk_scores > self.conf_thresh
            scores = topk_scores[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
            labels = topk_idxs % self.num_classes

            reg_pred_i = reg_pred_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]

            # decode box: [M, 4]
            bboxes = self.decode_boxes(anchors_i, reg_pred_i, self.stride[level])

            all_scores.append(scores)
            all_labels.append(labels)
            all_bboxes.append(bboxes)

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        bboxes = torch.cat(all_bboxes)

        # to cpu
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # nms
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # 归一化边界框
        bboxes = bboxes / self.input_size
        bboxes = np.clip(bboxes, 0., 1.)

        return bboxes, scores, labels


    @torch.no_grad()
    def inference(self, x):
        # backbone
        feats = self.backbone(x)
        c3, c4, c5 = feats['c3'], feats['c4'], feats['c5']

        # FPN, 多尺度特征融合
        p5 = self.conv_set_3(c5)
        p5_up = F.interpolate(self.conv_1x1_3(p5), scale_factor=2.0, mode='bilinear', align_corners=True)

        p4 = torch.cat([c4, p5_up], 1)
        p4 = self.conv_set_2(p4)
        p4_up = F.interpolate(self.conv_1x1_2(p4), scale_factor=2.0, mode='bilinear', align_corners=True)

        p3 = torch.cat([c3, p4_up], 1)
        p3 = self.conv_set_1(p3)

        # head
        # s = 32, 预测大物体
        p5 = self.extra_conv_3(p5)
        pred_3 = self.pred_3(p5)

        # s = 16, 预测中物体
        p4 = self.extra_conv_2(p4)
        pred_2 = self.pred_2(p4)

        # s = 8, 预测小物体
        p3 = self.extra_conv_1(p3)
        pred_1 = self.pred_1(p3)

        preds = [pred_1, pred_2, pred_3]
        conf_preds = []
        cls_preds = []
        txtytwth_preds = []

        B = x.size(0)
        KA = self.num_anchors
        NC = self.num_classes

        for pred in preds:
            # 对pred 的size做一些view调整，便于后续的处理
            # [B, KA * C, H, W] -> [B, H, W, KA * C] -> [B, H*W, KA*C]
            pred = pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

            # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测   
            # [B, H*W, KA*C] -> [B, H*W, KA] -> [B, H*W*KA, 1]
            conf_pred = pred[..., :1*KA].contiguous().view(B, -1, 1)
            # [B, H*W, KA*C] -> [B, H*W, KA*NC] -> [B, H*W*KA, NC]
            cls_pred = pred[:, :, 1*KA : (1+NC)*KA].contiguous().view(B, -1, NC)
            # [B, H*W, KA*C] -> [B, H*W, KA*4] -> [B, H*W*KA, 4]
            txtytwth_pred = pred[:, :, (1+NC)*KA:].contiguous().view(B, -1, 4)

            conf_pred = conf_pred[0]
            cls_pred = cls_pred[0]
            txtytwth_pred = txtytwth_pred[0]

            conf_preds.append(conf_pred)
            cls_preds.append(cls_pred)
            txtytwth_preds.append(txtytwth_pred)

        # 后处理
        bboxes, scores, labels = self.postprocess(conf_preds, cls_preds, txtytwth_preds)

        return bboxes, scores, labels
                       

    def forward(self, x, targets=None):
        if not self.trainable:
            return self.inference(x)
        else:
            # backbone
            feats = self.backbone(x)
            c3, c4, c5 = feats['c3'], feats['c4'], feats['c5']

            # FPN, 多尺度特征融合
            p5 = self.conv_set_3(c5)
            p5_up = F.interpolate(self.conv_1x1_3(p5), scale_factor=2.0, mode='bilinear', align_corners=True)

            p4 = torch.cat([c4, p5_up], 1)
            p4 = self.conv_set_2(p4)
            p4_up = F.interpolate(self.conv_1x1_2(p4), scale_factor=2.0, mode='bilinear', align_corners=True)

            p3 = torch.cat([c3, p4_up], 1)
            p3 = self.conv_set_1(p3)

            # head
            # s = 32, 预测大物体
            p5 = self.extra_conv_3(p5)
            pred_3 = self.pred_3(p5)

            # s = 16, 预测中物体
            p4 = self.extra_conv_2(p4)
            pred_2 = self.pred_2(p4)

            # s = 8, 预测小物体
            p3 = self.extra_conv_1(p3)
            pred_1 = self.pred_1(p3)

            preds = [pred_1, pred_2, pred_3]
            conf_preds = []
            cls_preds = []
            txtytwth_preds = []
            x1y1x2y2_preds = []

            B = x.size(0)
            KA = self.num_anchors
            NC = self.num_classes

            for level, pred in enumerate(preds):
                # 对pred 的size做一些view调整，便于后续的处理
                # [B, KA * C, H, W] -> [B, H, W, KA * C] -> [B, H*W, KA*C]
                pred = pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

                # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测   
                # [B, H*W, KA*C] -> [B, H*W, KA] -> [B, H*W*KA, 1]
                conf_pred = pred[..., :1*KA].contiguous().view(B, -1, 1)
                # [B, H*W, KA*C] -> [B, H*W, KA*NC] -> [B, H*W*KA, NC]
                cls_pred = pred[:, :, 1*KA : (1+NC)*KA].contiguous().view(B, -1, NC)
                # [B, H*W, KA*C] -> [B, H*W, KA*4] -> [B, H*W*KA, 4]
                txtytwth_pred = pred[:, :, (1+NC)*KA:].contiguous().view(B, -1, 4)
                # 解算边界框
                x1y1x2y2_pred = self.decode_boxes(self.anchor_boxes[level], txtytwth_pred, self.stride[level])
                x1y1x2y2_pred = x1y1x2y2_pred / self.input_size

                conf_preds.append(conf_pred)
                cls_preds.append(cls_pred)
                txtytwth_preds.append(txtytwth_pred)
                x1y1x2y2_preds.append(x1y1x2y2_pred)

            # 将所有结果沿着H*W这个维度拼接
            conf_pred = torch.cat(conf_preds, dim=1)
            cls_pred = torch.cat(cls_preds, dim=1)
            txtytwth_pred = torch.cat(txtytwth_preds, dim=1)
            x1y1x2y2_pred = torch.cat(x1y1x2y2_preds, dim=1)

            # 计算pred box与gt box之间的IoU
            x1y1x2y2_pred = x1y1x2y2_pred.view(-1, 4)
            x1y1x2y2_gt = targets[:, :, 7:].view(-1, 4)
            iou_pred = iou_score(x1y1x2y2_pred, x1y1x2y2_gt).view(B, -1, 1)

            # gt conf，这一操作是保证iou不会回传梯度
            with torch.no_grad():
                gt_conf = iou_pred.clone()

            # 我们讲pred box与gt box之间的iou作为objectness的学习目标. 
            # [obj, cls, txtytwth, scale_weight, x1y1x2y2] -> [conf, obj, cls, txtytwth, scale_weight]
            targets = torch.cat([gt_conf, targets[:, :, :7]], dim=2)

            # 计算loss
            (
                conf_loss,
                cls_loss,
                bbox_loss,
                total_loss
            ) = compute_loss(pred_conf=conf_pred, 
                                pred_cls=cls_pred,
                                pred_txtytwth=txtytwth_pred,
                                targets=targets
                                )

            return conf_loss, cls_loss, bbox_loss, total_loss 
                        