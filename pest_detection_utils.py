import numpy as np
import torch
import cv2
import random
import argparse
from detectron2.config import get_cfg
from Detic.third_party.CenterNet2.centernet.config import add_centernet_config
from Detic.detic.config import add_detic_config
import torchvision


def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE = "cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS",
                "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"],
        nargs=argparse.REMAINDER,
    )
    
    parser.add_argument('--nms_max_overlap', type=float, default=0.7,
                        help='Non-maxima suppression threshold: Maximum detection overlap.')
    
    parser.add_argument("--debug", help="Path to debug folder.", default=None)
    return parser


def non_max_suppression(prediction, iou_thres=0.45):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes
    Returns:
         Dict of predictions with the not overlapped instances 
    """
    bboxes, confs, clss, masks = detic2flat(prediction) #bboxes in xyxy
    idx = torchvision.ops.nms(bboxes, confs, iou_thres)  # NMS
    idx_np=idx.cpu().numpy()
    keep_predictions = {}
    keep_predictions['instances'] = prediction['instances'][idx_np]
    return keep_predictions


def detic2flat(predictions):
    if "instances" in predictions:
        bboxes = predictions["instances"].pred_boxes.tensor  # boxes in x1y1x2y2 format
        confs = predictions["instances"].scores
        clss = predictions["instances"].pred_classes
        masks = predictions["instances"].pred_masks
        return bboxes, confs, clss, masks
    else:
        return [], [], [], []


#   torch.jit.script
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height

    return y


def xyxy2tlwh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x1, y1, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0]            # x min
    y[:, 1] = x[:, 1]            # y min
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height

    return y


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def get_color_for(class_num):
    colors = [
        "#4892EA",
        "#00EEC3",
        "#FE4EF0",
        "#F4004E",
        "#FA7200",
        "#EEEE17",
        "#90FF00",
        "#78C1D2",
        "#8C29FF"
    ]

    num = hash(class_num) # may actually be a number or a string
    hex = colors[num%len(colors)]

    # adapted from https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
    rgb = tuple(int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    return rgb
    
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img
        
def draw_detections(img, prediction, clss_name):
    img_draw = np.copy(img).astype("float")
    bboxes, confs, clss, masks = detic2flat(prediction) #bboxes in xyxy
    for ii, bbox in enumerate(bboxes):
        box = bbox.cpu().numpy().astype('int')
        label_box =  str(clss_name[clss[ii].cpu().numpy()]) + " " + str(np.round(confs[ii].cpu().numpy(),2))
        img_draw = plot_one_box(box,img_draw,label=label_box)
    return img_draw

def det2img(img, predictions_nms, classes_names):
    # Draw detections over image and save it 
    out_img = draw_detections(img, predictions_nms, classes_names)
    
    # Generate masks
    bboxes, confs, clss, masks = detic2flat(predictions_nms) #bboxes in xyxy
    
    total_mask = np.zeros((masks.shape[1], masks.shape[2]))
    for jj in range(len(masks)):
        total_mask = total_mask + masks[jj].cpu().numpy().astype(int)
    if np.max(total_mask)>1:
        total_mask[total_mask > 1] = 1
        
    out_img_masked = cv2.bitwise_and(img, img,  mask=total_mask.astype("uint8"))
    
    return out_img, out_img_masked, total_mask