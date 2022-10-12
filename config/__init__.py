from .yolov3_config import yolov3_config


def build_model_config(args):
    return yolov3_config[args.version]