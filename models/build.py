from .yolov3 import YOLOv3


def build_yolov3(args, cfg, device, input_size, num_classes=20, trainable=False):
    anchor_size = cfg['anchor_size'][args.dataset]
    
    model = YOLOv3(
        cfg=cfg,
        device=device,
        input_size=input_size,
        num_classes=num_classes,
        trainable=trainable,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        topk=args.topk,
        anchor_size=anchor_size
        )

    return model
