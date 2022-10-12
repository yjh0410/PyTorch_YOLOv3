from .darknet53 import build_darknet53


def build_backbone(model_name='darknet53', pretrained=False):
    if model_name == 'darknet53':
        backbone, feat_dims = build_darknet53(pretrained)

    return backbone, feat_dims
