import numpy as np
import torch
import cv2
import random
import argparse
from detectron2.config import get_cfg
from Detic.third_party.CenterNet2.centernet.config import add_centernet_config
from Detic.detic.config import add_detic_config
import torchvision
import os
import time


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

    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg' or a path to a video",
    )
    parser.add_argument(
        "--output",
        default="output/",
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
                "models/Detic_LI_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"],
        nargs=argparse.REMAINDER,
    )
    
    parser.add_argument('--nms_max_overlap', type=float, default=0.7,
                        help='Non-maxima suppression threshold: Maximum detection overlap.')
    
    parser.add_argument("--debug", default=False, action='store_true', help="Path to debug folder.")
    parser.add_argument("--patching", default=False, action='store_true', help="To proccess the image in patches.")
    parser.add_argument("--patch_size", default=640, type=int, help="Patch of the patches.")
    parser.add_argument("--overlap", default=0.2, type=float, help="Overlap of the patches")
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
    
def detic_post_proccess(img, detic_predictor, predictions, args):
    predictions_nms = non_max_suppression(predictions, iou_thres=args.nms_max_overlap)
   
    pred_str = "detected {} instances".format(len(predictions_nms["instances"]))
    
    classes_names = detic_predictor.metadata.thing_classes
    
    if len(predictions_nms['instances']) != 0:
        img_out_bbox, img_out_mask, mask_tot = detic2img(img, predictions_nms, classes_names)
    else: 
        img_out_bbox = img
        img_out_mask = np.zeros((img.shape[0], img.shape[1]))
        mask_tot = np.zeros((img.shape[0], img.shape[1]))
    
    return img_out_bbox, img_out_mask, mask_tot, pred_str, predictions_nms


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


def generate_final_mask(masks, img, fruit_zone=(0,0,0,0)):
    total_mask = np.zeros((img.shape[0], img.shape[1]))
    if fruit_zone == (0,0,0,0):
        fruit_zone = (0, 0, img.shape[0], img.shape[1])

    for jj in range(len(masks)):
        total_mask[fruit_zone[0]:fruit_zone[2], fruit_zone[1]:fruit_zone[3]] = total_mask[fruit_zone[0]:fruit_zone[2], fruit_zone[1]:fruit_zone[3]] + masks[jj].cpu().numpy().astype(int)[0:(fruit_zone[2]-fruit_zone[0]), 0:(fruit_zone[3]-fruit_zone[1])]
        #total_mask = total_mask + masks[jj].cpu().numpy().astype(int)[0:total_mask.shape[0], 0:total_mask.shape[1]]
    if np.max(total_mask)>1:
        total_mask[total_mask > 1] = 1
        
    out_img_masked = cv2.bitwise_and(img, img,  mask=total_mask.astype("uint8"))
    return out_img_masked, total_mask


        
def draw_detections(img, prediction, clss_name):
    img_draw = np.copy(img).astype("float")
    bboxes, confs, clss, masks = prediction #bboxes in xyxy
    for ii, bbox in enumerate(bboxes):
        box = bbox.cpu().numpy().astype('int')
        label_box =  str(clss_name[int(clss[ii].cpu().numpy())]) + " " + str(np.round(confs[ii].cpu().numpy(),2))
        img_draw = plot_one_box(box,img_draw,label=label_box)
    return img_draw

def lists2img(img, predictions_nms, classes_names, fruit_zone):
    bboxes, confs, clss, masks = predictions_nms
    # Draw detections over image and save it 
    out_img = draw_detections(img, predictions_nms, classes_names)
    
    # Generate masks
    out_img_masked, total_mask = generate_final_mask(masks, img, fruit_zone)
    
    return out_img, out_img_masked, total_mask


def detic2img(img, predictions_nms, classes_names):
    predictions_nms  = detic2flat(predictions_nms) #bboxes in xyxy
    bboxes, confs, clss, masks = predictions_nms 
    # Draw detections over image and save it 
    out_img = draw_detections(img, predictions_nms, classes_names)
    
    # Generate masks
    out_img_masked, total_mask = generate_final_mask(masks, img)
    
    return out_img, out_img_masked, total_mask


def patch_detic2list(empty_mask, detic_pred, patch_size, row, column, overlap, fruit_zone):
    bboxes, confs, clss, masks= detic2flat(detic_pred)
    # bboxes to img coordinates
    for ii in range(len(bboxes)):
        bboxes[ii][0] = bboxes[ii][0] + column*patch_size*(1 - overlap) + fruit_zone[1]
        bboxes[ii][1] = bboxes[ii][1] + row*patch_size*(1 - overlap) + fruit_zone[0]
        bboxes[ii][2] = bboxes[ii][2] + column*patch_size*(1 - overlap) + fruit_zone[1]
        bboxes[ii][3] = bboxes[ii][3] + row*patch_size*(1 - overlap) + fruit_zone[0]
    # masks to img coordinates
    total_mask = np.zeros((empty_mask.shape[0], empty_mask.shape[1]))
    for jj in range(len(masks)):
        total_mask_patch = total_mask[round(row*patch_size*(1 - overlap)): round(row*patch_size*(1 - overlap) + patch_size), round(column*patch_size*(1 - overlap)):round(column*patch_size*(1 - overlap) + patch_size)]
        window_shape = total_mask_patch.shape
        if total_mask_patch.shape != (patch_size,patch_size):
                total_mask_patch = cv2.copyMakeBorder(total_mask_patch,0,int(patch_size-total_mask_patch.shape[0]), 0, int(patch_size-total_mask_patch.shape[1]),cv2.BORDER_CONSTANT, value=[0, 0, 0])
        total_mask_patch = total_mask_patch + masks[jj].cpu().numpy().astype(int)
        if total_mask_patch.shape != window_shape:
                total_mask_patch = total_mask_patch[0:window_shape[0], 0:window_shape[1]]
        total_mask[round(row*patch_size*(1 - overlap)):round(row*patch_size*(1 - overlap) + patch_size), round(column*patch_size*(1 - overlap)):round(column*patch_size*(1 - overlap) + patch_size)] = total_mask_patch
    if np.max(total_mask)>1:
        total_mask[total_mask > 1] = 1

    return bboxes, confs, clss, masks, total_mask
'''
def detic_proccess_img_old(img_p, args, detic_predictor, logger, save=True, save_path="", path="", fruit_zone=(0,0,0,0), img_o=None):
    if img_o is None:
        img_o=img_p
    
    if args.patching:
        patch_size = args.patch_size
        overlap = args.overlap
        patches, empty_mask = im2patches(img_p, patch_size, overlap)
        n_row, n_col, _, _, _ = patches.shape
        logger.info("{} patches: {} rows and {} columns".format(str(n_row*n_col), str(n_row), str(n_col)))
        bboxs_t, confs_t, clss_t, masks_t = ([] for i in range(4))
        for ii in range(n_row):
            for jj in range(n_col):
                start_time = time.time()
                img_patch = patches[ii][jj]
                pred_patch, visualized_output = detic_predictor.run_on_image(img_patch)
                img_out_bbox, img_out_mask, mask_tot, pred_str, p_patch_nms= detic_post_proccess(img_patch, detic_predictor, pred_patch, args)
                
                # save detections in full img coordinates
                bboxs_p, confs_p, clss_p, masks_p = patch_detic2list(empty_mask, p_patch_nms, patch_size, ii, jj, overlap, fruit_zone)
                bboxs_t = bboxs_t + bboxs_p.tolist()
                confs_t = confs_t + confs_p.tolist()
                clss_t = clss_t + clss_p.tolist()
                masks_t.append(masks_p) 
                
                logger.info("{} {}: {} in {:.2f}s".format(path, "patch " + str(ii) + " " + str(jj) ,pred_str, time.time() - start_time)) 
                    
                if save:
                    if save_path != "":
                        if os.path.isdir(save_path):
                            assert os.path.isdir(save_path), save_path
                            out_filename = os.path.join(save_path, os.path.basename(path))
                        else:
                            assert len(args.input) == 1, "Please specify a directory with save_path"
                            out_filename = save_path
                    else:
                        out_filename=""
                    
                    cv2.imwrite(out_filename + "_p_" + str(ii) + str(jj) +".jpg", img_patch)
                    cv2.imwrite(out_filename + "_p_" + str(ii) + str(jj) +"_vis.jpg", img_out_bbox)
                    cv2.imwrite(out_filename + "_p_" + str(ii) + str(jj) +"_mask.jpg", img_out_mask)  
                
        pred_compose = torch.FloatTensor(np.array(bboxs_t)), torch.FloatTensor(np.array(confs_t)), torch.FloatTensor(np.array(clss_t)), torch.FloatTensor(np.array(masks_t))
        if len(pred_compose[0]):
            pred_compose_nms = non_max_suppression_compose(pred_compose, iou_thres=args.nms_max_overlap)
        else:
            pred_compose_nms = pred_compose
        classes_names = detic_predictor.metadata.thing_classes
        img_out_bbox, img_out_mask, mask_tot = lists2img(np.asarray(img_o), pred_compose_nms, classes_names, fruit_zone)
        
        if save:
            if save_path:
                if os.path.isdir(save_path):
                    assert os.path.isdir(save_path), save_path
                    out_filename = os.path.join(save_path, os.path.basename(path))
                else:
                    assert len(save_path) == 1, "Please specify a directory with args.output"
                    out_filename = save_path
            else:
                out_filename=""
            
            cv2.imwrite(out_filename +"_vis.jpg", img_out_bbox)
            cv2.imwrite(out_filename +"_mask.jpg", img_out_mask)    
            
    else:
        start_time = time.time()
        predictions, visualized_output = detic_predictor.run_on_image(img_p)
        img_out_bbox, img_out_mask, mask_tot, pred_str, _= detic_post_proccess(img_p, detic_predictor, predictions, args)
        logger.info("{}: {} in {:.2f}s".format(path, pred_str, time.time() - start_time)) 
        
        if save:
            if save_path:
                if os.path.isdir(save_path):
                    assert os.path.isdir(save_path),save_path
                    out_filename = os.path.join(save_path, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with save_path"
                    out_filename = save_path
            else:
                out_filename=""
            
            cv2.imwrite(out_filename +"_vis.jpg", img_out_bbox)
            cv2.imwrite(out_filename +"_mask.jpg", img_out_mask)  
            
    return img_out_bbox, img_out_mask, mask_tot
'''

def detic_single_im(args, detic_predictor, logger, save=True, save_path="",img_processed=None, fruit_zone=(0,0,0,0), img_original=None):
    if img_processed is not None:
        img_out_bbox, img_out_mask, mask = detic_proccess_img(img_processed, args, detic_predictor, logger, save=save, save_path=save_path, fruit_zone=fruit_zone, img_o=img_original)
    return img_out_bbox, img_out_mask, mask  
        
def detic_proccess_img(img, detic_predictor, args, fruit_zone, empty_mask, patch_id = (0,0)):
    # Predict
    pred_patch, visualized_output = detic_predictor.run_on_image(img) #TODO extract visualization part
    img_out_bbox, img_out_mask, mask_tot, pred_str, p_patch_nms= detic_post_proccess(img, detic_predictor, pred_patch, args)
    
    # Save detections in full img coordinates
    if args.patching:
        bboxs, confs, clss, masks, mask_agg = patch_detic2list(empty_mask, p_patch_nms, args.patch_size, patch_id[0], patch_id[1], args.overlap, fruit_zone)
    else:
        bboxs, confs, clss, masks = detic2flat(p_patch_nms)
        mask_agg = mask_tot
    return ((bboxs, confs, clss, masks, mask_agg), img_out_bbox, img_out_mask, mask_tot, pred_str)
              