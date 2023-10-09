import argparse
from detectron2.config import get_cfg
from Detic.third_party.CenterNet2.centernet.config import add_centernet_config
from Detic.detic.config import add_detic_config
from xdcoder_utils import load_config_dict_to_opt, load_opt_from_config_files, load_opt_command
import json
import random
import time
import torchvision
import logging
from scipy import stats
from Detic.detic.predictor import VisualizationDemo
from GSAM_utils import grdSAM_process_image

import os
import torch

from PIL import Image
import numpy as np
np.random.seed(27)

from torchvision import transforms

from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_colors
from detectron2.utils.colormap import random_color
from X_Decoder.xdecoder.BaseModel import BaseModel
from X_Decoder.xdecoder import build_model
from X_Decoder.utils.distributed import init_distributed

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, distance_transform_edt
import numpy as np
from skimage import measure
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon
from skimage.draw import polygon2mask

from detic_utils import detic_proccess_img
from yolo_utils import yolo_proccess_img
from scipy.stats import mode
from scipy import ndimage

def load_detic(args):
    cfg = setup_cfg_detic(args)

    # Instance Detic Predictor
    try:
        detic_predictor = VisualizationDemo(cfg, args)
    except:
        # second time it works
        detic_predictor = VisualizationDemo(cfg, args)
        logging.warning("w: 'CustomRCNN' was already registered")
        
    return detic_predictor

def load_xdecoder_refseg(args):
    
    opt, cmdline_args= setup_cfg_xdecoder(args)
    opt = init_distributed(opt)

    vocabulary_xdec = args.vocabulary_xdec

    model = BaseModel(opt, build_model(opt)).from_pretrained(args.xdec_pretrained_pth).eval().cuda()
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(["background", "background"], is_eval=True)

    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)

    metadata = MetadataCatalog.get('ade20k_panoptic_train')
    model.model.metadata = metadata
    
    return model, transform, metadata, vocabulary_xdec

def load_xdecoder_semseg(args):
    
    opt, cmdline_args= setup_cfg_xdecoder(args)
    opt = init_distributed(opt)

    vocabulary_xdec = args.vocabulary_xdec

    model = BaseModel(opt, build_model(opt)).from_pretrained(args.xdec_pretrained_pth).eval().cuda()
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(["background", "background"], is_eval=True)

    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)

    metadata = MetadataCatalog.get('ade20k_panoptic_train')
    model.model.metadata = metadata

    stuff_classes = vocabulary_xdec
    stuff_colors = [random_color(rgb=True, maximum=255).astype(np.int32).tolist() for _ in range(len(stuff_classes))]
    stuff_dataset_id_to_contiguous_id = {x:x for x in range(len(stuff_classes))}

    MetadataCatalog.get("demo").set(
        stuff_colors=stuff_colors,
        stuff_classes=stuff_classes,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
    )
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(stuff_classes + ["background"], is_eval=True)
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(stuff_classes)
    
    return model, transform, metadata, vocabulary_xdec

def setup_cfg_detic(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE = "cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file_detic)
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

def setup_cfg_xdecoder(args):
    cmdline_args = args

    opt = load_opt_from_config_files(cmdline_args.config_file_xdec)

    if cmdline_args.config_overrides:
        config_overrides_string = ' '.join(cmdline_args.config_overrides)
        logging.warning(f"Command line config overrides: {config_overrides_string}")
        config_dict = json.loads(config_overrides_string)
        load_config_dict_to_opt(opt, config_dict)

    if cmdline_args.overrides:
        assert len(cmdline_args.overrides) % 2 == 0, "overrides arguments is not paired, required: key value"
        keys = [cmdline_args.overrides[idx*2] for idx in range(len(cmdline_args.overrides)//2)]
        vals = [cmdline_args.overrides[idx*2+1] for idx in range(len(cmdline_args.overrides)//2)]
        vals = [val.replace('false', '').replace('False','') if len(val.replace(' ', '')) == 5 else val for val in vals]

        types = []
        for key in keys:
            key = key.split('.')
            ele = opt.copy()
            while len(key) > 0:
                ele = ele[key.pop(0)]
            types.append(type(ele))
        
        config_dict = {x:z(y) for x,y,z in zip(keys, vals, types)}
        load_config_dict_to_opt(opt, config_dict)

    # combine cmdline_args into opt dictionary
    for key, val in cmdline_args.__dict__.items():
        if val is not None:
            opt[key] = val

    return opt, cmdline_args


def get_parser():
    
    parser = argparse.ArgumentParser(description="Xdcoder and Detectron2 setup for builtin configs")
    
    # COMMON
    parser.add_argument(
        "--input",
        default="",
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
    
    parser.add_argument("--debug", default=False, action='store_true', help="Bool indicating if debug")
    parser.add_argument("--print_health", default=False, action='store_true', help="Bool indicating if print health over image")
    parser.add_argument("--full_pipeline", default=True, action='store_true', help="Bool indicating if use full pipeline approach")
    
    # DETIC SETUP    
    parser.add_argument(
        "--config-file-detic",
        default="Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--cpu", default=False, action='store_true', help="Use CPU only.")

    
    parser.add_argument(
        "--vocabulary",
        default="custom",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="fly,mosquito,buterfly,moth,ladybug,gnat",
        help="",
    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.2,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS",
                "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"],
        nargs=argparse.REMAINDER,
    )
    
    parser.add_argument('--nms_max_overlap', type=float, default=0.01,
                        help='Non-maxima suppression threshold: Maximum detection overlap.')
    
    
    parser.add_argument("--patching", default=True, action='store_true', help="To proccess the image in patches.")
    parser.add_argument("--patch_size", default=640, type=int, help="Patch of the patches.")
    parser.add_argument("--overlap", default=0.3, type=float, help="Overlap of the patches")
    parser.add_argument("--seg_step", default=True, action='store_true', help="To proccess the image in patches.")
    
    # XDECODER SETUP 
    
    parser.add_argument('--command', default="evaluate", help='Command: train/evaluate/train-and-evaluate')
    parser.add_argument('--config_file_xdec', default=['X_Decoder/configs/xdecoder/svlp_focalt_lang.yaml'], nargs='+', help='Path(s) to the config file(s).')
    parser.add_argument('--user_dir', help='Path to the user defined module for tasks (models, criteria), optimizers, and lr schedulers.')
    parser.add_argument('--config_overrides', nargs='*', help='Override parameters on config with a json style string, e.g. {"<PARAM_NAME_1>": <PARAM_VALUE_1>, "<PARAM_GROUP_2>.<PARAM_SUBGROUP_2>.<PARAM_2>": <PARAM_VALUE_2>}. A key with "." updates the object in the corresponding nested dict. Remember to escape " in command line.')
    parser.add_argument('--overrides', help='arguments that used to override the config file in cmdline', nargs=argparse.REMAINDER)
    parser.add_argument('--xdec_pretrained_pth', default='X_Decoder/models/xdecoder_focalt_last.pt', help='Path(s) to the weight file(s).')
    parser.add_argument('--xdec_img_size', type=int, default=1024 ,help='reshape size for the image to be proccessed wit x-decoder')
    parser.add_argument('--vocabulary_xdec', nargs='+', default=["yellow_trap","red_trap","blue_trap","ground","grass","vegetation","wood","sky","stick","bush","background"], help='Concepts to segmentate')
    parser.add_argument('--det_model', default="grdSAM", help='Select the model use for detection: Detic or YOLO')
    
    # GRDSAM
     # Classes for object detection
    parser.add_argument(
        "--classes_grdSAM",
        default=["leaf","insect","vegetation","dust","plant","hand","red_trap","yellow_trap","grass","weed","rope","hole","wire"],
        nargs='+',
        help="List of classes to detect."
    )
    # Classes for object detection hl
    parser.add_argument(
        "--classes_grdSAM_hl",
        default=["insect"],
        nargs='+',
        help="List of classes to detect and hl."
    )

    # Thresholds
    parser.add_argument(
        "--box_threshold_grdSAM",
        type=float,
        default=0.3,
        help="Box threshold for GroundingDINO."
    )
    parser.add_argument(
        "--text_threshold_grdSAM",
        type=float,
        default=0.3,
        help="Text threshold for GroundingDINO."
    )
    parser.add_argument(
        "--nms_threshold_grdSAM",
        type=float,
        default=0.7,
        help="Non-maximum suppression threshold."
    )

    parser.add_argument(
        "--save_imgs",
        type=bool,
        default=False,
        help="Save resulting images"
    )


    return parser


def non_max_suppression_compose(prediction, iou_thres=0.45):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes
    Returns:
         Dict of predictions with the not overlapped instances 
    """
    bbxs_out, confs_out, clss_out, masks_out = [], [], [], []
    bboxes, confs, clss, masks = prediction #bboxes in xyxy
    masks_out = masks
    
    idx = torchvision.ops.nms(bboxes, confs, iou_thres)  # NMS
    idx_np=idx.cpu().numpy()
    for idx in idx_np:
        bbxs_out.append(bboxes[idx]) 
        confs_out.append(confs[idx])
        clss_out.append(clss[idx]) 
    keep_predictions = bbxs_out, confs_out, clss_out, masks_out
    return keep_predictions


def kmeans(mask, max_clusters):
    points_array = np.argwhere(mask == 1)
    
    # Si no se especifica el número máximo de clusters, usar un valor por defecto
    if max_clusters is None:
        max_clusters = 10
    
    # Calcular el número óptimo de clusters para K-means.
    points_array = np.float32(points_array)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    dists = []
    for k in range(1, max_clusters):
        compactness, labels, centers = cv2.kmeans(points_array, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        dists.append(compactness)
    
    # Create figure and axes
    fig_kopt, ax = plt.subplots()
    # Generate 2D plot
    ax.plot(dists)
    # Configure axes
    ax.set_xlabel('N clusters')
    ax.set_ylabel('Value')
    ax.set_title('kmeans values')
    
    elbow_2 = np.gradient(np.gradient(dists))
    elbow_1 = np.gradient(dists)
    ax.plot(elbow_1)
    ax.plot(elbow_2)
    optimal_k = elbow_2.argmax() + 1
    print("num_cluster: ", str(optimal_k))

    # Aplicar K-means en la imagen para encontrar clusters de objetos segmentados.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(points_array, optimal_k, None, criteria, 10, flags)

    # Asignar a cada pixel el valor correspondiente al centroide más cercano.
    centers = np.uint8(centers)
    labels = labels.flatten()
    img_clusters = np.zeros((mask.shape[0], mask.shape[1], 3))
    colors = random_colors(k)
    
    for ii in range(points_array.shape[0]):
        img_clusters[int(points_array[ii][0])][int(points_array[ii][1])] = colors[labels[ii]]
        
    return img_clusters, k, fig_kopt


def process_mask(mask, save=False, save_path=None, max_clusters=20):
    # Apply dilation to close gaps between segmented instances.
    kernel_op = np.ones((10, 10),np.uint8)
    kernel_b = np.ones((5, 5),np.uint8)
    
    # Apply opening to reduce noise and smooth the edges of segmented instances.
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_op)
    
    dilated_mask = cv2.dilate(closed_mask, kernel_b, iterations=3)
    
    # Apply clossing to avoid gaps.
    opened_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_OPEN, kernel_op)
    
    # Erode to quit extra margins
    proccessed_mask = cv2.erode(opened_mask, kernel_b, iterations=3)
    
    #kmeans_centroids_img, num_c, fig_optK = kmeans(proccessed_mask, max_clusters)
    
    if save:
        #cv2.imwrite(os.path.join(save_path,"dilated_mask.png"), dilated_mask*255)
        cv2.imwrite(os.path.join(save_path,"closed_mask.png"), closed_mask*255)
        cv2.imwrite(os.path.join(save_path,"opened_mask.png"), opened_mask*255)
        #cv2.imwrite(os.path.join(save_path,"kmeans.png"), kmeans_centroids_img)
        # Save plot to disk
        #fig_optK.savefig(os.path.join(save_path,"k_values.png"))
    
    #num_clusters = count_contiguous_groups(proccessed_mask)
    #print("num insects detected: " + str(num_clusters))
    
    return proccessed_mask

def count_contiguous_groups(mask):
    # Convert the mask to boolean type (if it is not already)
    mask = np.array(mask, dtype=bool)

    # Get the contiguous groups of pixels with value 1 in the mask
    groups, count = label(mask)

    # Return the number of contiguous groups found
    return count


def mask_metrics(mask_true, mask_pred):
    """Compute the Dice coefficient, Jaccard index, and Hausdorff distance for two binary masks."""
    
    # Ensure that the input masks have the same size
    if mask_pred.shape != mask_true.shape:
        raise ValueError('Input masks must have the same size.')
    
    # Compute the Dice coefficient and Jaccard index
    intersection = np.logical_and(mask_true, mask_pred)
    union = np.logical_or(mask_true, mask_pred)
    iou = intersection.sum() / union.sum()
    dice = 2 * intersection.sum() / (mask_true.sum() + mask_pred.sum())
    jaccard = intersection.sum() / (mask_true.sum() + mask_pred.sum() - intersection.sum())
    
    # Compute the Hausdorff distance
    distances_true = distance_transform_edt(mask_true)
    distances_pred = distance_transform_edt(mask_pred)
    hausdorff = np.max([np.percentile(distances_true[distances_pred > 0], 95),
                        np.percentile(distances_pred[distances_true > 0], 95)])
    
    # Return the metrics
    return iou, dice, jaccard, hausdorff


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

def bbox_to_coco(bbox, img_size):
    x_min, y_min, x_max, y_max = bbox
    img_width, img_height = img_size

    # Convert to relative coords
    x_min_rel = x_min / img_width
    y_min_rel = y_min / img_height
    x_max_rel = x_max / img_width
    y_max_rel = y_max / img_height

    return [round(x_min_rel,4), round(y_min_rel,4), round(x_max_rel,4), round(y_max_rel,4)]


def pred2COCOannotations(img, mask_final, predictions_nms, out_folder="", file_name=""):
    if len(predictions_nms) == 4:
        bboxes, confs, clss, masks = predictions_nms
    else:
        bboxes, confs, clss, _ , masks = predictions_nms

    height, width, _ = img.shape
    
    txt_path_file = os.path.join(out_folder, file_name +".txt")

    coco_annotations = {
        "info": {
            "description": "COCO-style annotations",
            "version": "1.0"
        },
        "images": [
            {
                "id": out_folder,
                "width": width,
                "height": height,
                }
        ],
        "detections": [],
        "segmentations": [],
    }
    if len(bboxes)>0:
        for i, (bbox, conf, cls, msk) in enumerate(zip(bboxes, confs, clss, masks)):
            h, w, _ = img.shape
            coco_bbox = bbox_to_coco(np.asarray(bbox.cpu()), (w,h))
            annotation = {
                "id": i + 1,
                "image_id": out_folder + file_name,
                "width": width,
                "height": height,
                "category_id": int(np.asarray(cls.cpu())),
                "bbox": str(coco_bbox),
                "score": round(float(np.asarray(conf.cpu())),2)
            }
            coco_annotations["detections"].append(annotation)
            
    # Generate segmentation annotation
    seg_annotations = mask_to_coco_segmentation(mask_final)
    coco_annotations["segmentations"].append(seg_annotations)

    if out_folder != "":
        with open(txt_path_file, "w") as outfile:
            annotations_json = json.dump(coco_annotations, outfile, indent=2)
    else:
        annotations_json = json.dumps(coco_annotations)
    return coco_annotations


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


def zoom_on_non_zero(image, black_threshold=0.8):
    # Convert the image to grayscale to facilitate non-zero pixel detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the percentage of black pixels
    total_pixels = gray.size
    black_pixels = np.count_nonzero(gray == 0)
    black_ratio = black_pixels / total_pixels

    if black_ratio > black_threshold:
        # Find the indices of the non-zero elements
        rows, cols = np.nonzero(gray)

        # Find the minimum and maximum coordinates of the non-zero rows and columns
        row_min, row_max = np.min(rows), np.max(rows)
        col_min, col_max = np.min(cols), np.max(cols)

        # Calculate the region of interest (ROI) using the minimum and maximum coordinates
        roi = image[row_min:row_max, col_min:col_max]

        # Calculate the desired non-black pixel percentage in the output image
        target_non_black_ratio = 1 - black_threshold

        # Calculate the scaling factor needed to achieve the desired non-black pixel percentage
        input_non_black_pixels = total_pixels - black_pixels
        roi_non_black_pixels = roi.size / 3 - np.count_nonzero(roi == 0) / 3
        scale_factor = np.sqrt(input_non_black_pixels / roi_non_black_pixels / target_non_black_ratio)

        # Resize the output image to have a non-black pixel percentage equal to the black_threshold value
        resized_roi = cv2.resize(roi, (int(roi.shape[1] * scale_factor), int(roi.shape[0] * scale_factor)), interpolation=cv2.INTER_LINEAR)

        # Create a black image of the same size as the original image
        zoomed_img = np.zeros_like(image)

        # Calculate the coordinates where the resized image will be placed in the output image
        start_row = (image.shape[0] - resized_roi.shape[0]) // 2
        start_col = (image.shape[1] - resized_roi.shape[1]) // 2

        # Place the resized image in the output image
        zoomed_img[start_row:start_row + resized_roi.shape[0], start_col:start_col + resized_roi.shape[1]] = resized_roi

        return zoomed_img
    else:
        return image


def patch_image(image, patch_size, overlap):
    image_tiles = []
    step = round(patch_size*(1-overlap))
    h, w, n_channels = image.shape
    if h == patch_size:
        step_h = patch_size
    else:
        step_h = step
    if w == patch_size:
        step_w = patch_size
    else:
        step_w = step
    
    for y in range(0, image.shape[0], step_h):
        for x in range(0, image.shape[1], step_w):
            image_tile = image[y:y + patch_size, x:x + patch_size]
            if image_tile.shape != (patch_size,patch_size, n_channels):
                image_tile= cv2.copyMakeBorder(image_tile,0,int(patch_size-image_tile.shape[0]), 0, int(patch_size-image_tile.shape[1]),cv2.BORDER_CONSTANT, value=[0, 0, 0])
            image_tiles.append(image_tile)
    return image_tiles
            
def im2patches(img, patch_size=640, overlap=0.2, max_patches=80):
    """
    Divide the input image into overlapping patches and groups them by rows and columns.
    
    Parameters:
    img (np.array): Input image in numpy array format.
    patch_size (int): Desired size for each square patch.
    overlap (float): Fraction of overlap between patches (0 to 1).
    max_patches (int): Maximum number of patches to be extracted.
    
    Returns:
    np.array: A numpy array containing the extracted image patches grouped by rows and columns.
    np.array: A 2D numpy array representing an empty mask with dimensions corresponding to the padded image.
    """
    
    # Obtain the dimensions of the image
    height, width = img.shape[:2]
    
    # Calculate the step between patches
    step = int(patch_size * (1 - overlap))
    
    # Calculate the number of steps required in x and y directions
    x_steps = list(range(0, width, step))
    y_steps = list(range(0, height, step))
    
    # Check if padding is necessary for the last patch
    if x_steps[-1] + patch_size > width:
        pad_right = x_steps[-1] + patch_size - width
    else:
        pad_right = 0
        
    if y_steps[-1] + patch_size > height:
        pad_bottom = y_steps[-1] + patch_size - height
    else:
        pad_bottom = 0
    
    # If padding is necessary, pad the image with zeros (black)
    if pad_bottom > 0 or pad_right > 0:
        img = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    # Initialize the empty_mask with zeros
    empty_mask = np.zeros((height + pad_bottom, width + pad_right), dtype=np.uint8)
    
    # Initialize the list of patches
    patches = []
    
    # Traverse the padded image extracting patches and grouping them by rows and columns
    for y in y_steps:
        row_patches = []
        for x in x_steps:
            # Extract the patch
            patch = img[y:y + patch_size, x:x + patch_size]
            row_patches.append(patch)
        patches.append(row_patches)
    
    # Limit the number of patches to max_patches if necessary
    patches_flat = [patch for row in patches for patch in row]
    if len(patches_flat) > max_patches:
        logging.info("too much patches")
        #patches_flat = patches_flat[:max_patches]
        #patches = np.array(patches_flat).reshape((-1, len(x_steps), *patch.shape))
    
    return np.array(patches), empty_mask


def check_health(health_model, img_patch, masks_p, health_thres):
    health_mask = np.zeros(img_patch.shape)
    health_flag = False
    for msk in masks_p:
        msk_np = msk.cpu().numpy().astype("uint8")
       
        if msk_np.shape != img_patch.shape:
            msk_np = cv2.resize(msk_np, (640,640)) # fix image to model input dimensions 
        img_crop_health = cv2.bitwise_and(img_patch, img_patch, mask=msk_np)    
        img_health_in = cv2.resize(img_crop_health, (640,640)) # fix image to model input dimensions
        img_zoomed = zoom_on_non_zero(img_health_in) 
        results = health_model(img_zoomed, imgsz=640)
        disease_score = float(results[0].probs[0].cpu().detach().numpy())
        healthy_score =float(results[0].probs[1].cpu().detach().numpy())
        if disease_score > health_thres:
            health_flag = True
            health_msg_d = "disease detected, conf: {:.2f}".format(disease_score)
            color = (0,0,255)
            color_mask = cv2.cvtColor(msk_np, cv2.COLOR_GRAY2RGB) * color
            health_mask = health_mask + color_mask
        else:
            health_msg_h ="healthy, conf: {:.2f}".format(healthy_score)
            color = (255,0,0)
            color_mask = cv2.cvtColor(msk_np, cv2.COLOR_GRAY2RGB) * color
            health_mask = health_mask + color_mask
    if health_flag:
        health_msg = health_msg_d
    else: 
        health_msg = health_msg_h
    txt_position = (10, health_mask.shape[0]-10)
    img_health = cv2.putText(np.copy(health_mask), health_msg, txt_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1) 
    
    return img_health, health_mask, health_msg, health_flag
  
  
def patchmask2imgmask(mask_agg, mask, row, column, patch_size, overlap, fruit_zone=(0,0,0,0)):
    # from patches mask to full dimension mask
    mask_agg_patch = mask_agg[round(row*patch_size*(1 - overlap)):round(row*patch_size*(1 - overlap) + patch_size), round(column*patch_size*(1 - overlap)):round(column*patch_size*(1 - overlap) + patch_size)]
    window_shape = mask_agg_patch.shape
    if mask_agg_patch.shape != (patch_size,patch_size,3):
            mask_agg_patch = cv2.copyMakeBorder(mask_agg_patch,0,int(patch_size-mask_agg_patch.shape[0]), 0, int(patch_size-mask_agg_patch.shape[1]),cv2.BORDER_CONSTANT, value=[0, 0, 0])
    mask_agg_patch = mask_agg_patch + mask
    if mask_agg_patch.shape != window_shape:
            mask_agg_patch = mask_agg_patch[0:window_shape[0], 0:window_shape[1]]
    mask_agg[(round(row*patch_size*(1 - overlap))):(round(row*patch_size*(1 - overlap) + patch_size)), (round(column*patch_size*(1 - overlap))):(round(column*patch_size*(1 - overlap) + patch_size))] = mask_agg_patch
    if np.max(mask_agg)>255:
        mask_agg[mask_agg > 255] = 255

    return mask_agg

def predict_img(img_p, args, model_predictor, save=True, save_path="", path="", fruit_zone=(0,0,0,0), img_o=None, cont_det=0):
    # If image to be proccessed is a patch of the original both have to be pass as input
    if img_o is None:
        img_o=img_p
    
    # Predict image patches   
    if args.patching:
        patch_size = args.patch_size
        overlap = args.overlap
        patches, empty_mask= im2patches(img_p, patch_size, overlap)
        n_row, n_col, _, _, _ = patches.shape
        logging.info("{} patches: {} rows and {} columns".format(str(n_row*n_col), str(n_row), str(n_col)))
        bboxs_t, confs_t, clss_t, masks_t = ([] for i in range(4))
        for ii in range(n_row):
            for jj in range(n_col):
                # Detect over each patch
                start_time = time.time()
                img_patch = patches[ii][jj]
                #simg_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB)
                if args.det_model == "Detic":
                    detections_p, img_out_bbox, img_out_mask, mask_tot, pred_str = detic_proccess_img(img_patch, model_predictor, args, fruit_zone, empty_mask, (ii, jj))
                    bboxs_p, confs_p, clss_p, masks_p, mask_p_agg = detections_p
                elif args.det_model == "grdSAM": 
                    detections_p, img_out_bbox, img_out_mask, mask_tot, pred_str = grdSAM_process_image(img_patch, model_predictor, args, fruit_zone, empty_mask, (ii, jj))
                    bboxs_p, confs_p, clss_p, masks_p, mask_p_agg = detections_p
                else:
                    detections_p, img_out_bbox, img_out_mask, mask_tot, pred_str = yolo_proccess_img(img_patch, model_predictor, args, fruit_zone, empty_mask, (ii, jj))
                    bboxs_p, confs_p, clss_p, masks_p, mask_p_agg = detections_p
                        
                bboxs_t = bboxs_t + bboxs_p.tolist()
                confs_t = confs_t + confs_p.tolist()
                clss_t = clss_t + clss_p.tolist()
                masks_t.append(mask_p_agg) 
                
                logging.info("{} {}: {} in {:.2f}s".format(path, "patch " + str(ii) + " " + str(jj) , pred_str, time.time() - start_time)) 
                
                # Save results   
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
                    
        # Join all the predictions over patches
        pred_compose = torch.FloatTensor(np.array(bboxs_t)), torch.FloatTensor(np.array(confs_t)), torch.FloatTensor(np.array(clss_t)), torch.FloatTensor(np.array(masks_t))
        if len(pred_compose[0]):
            pred_compose_nms = non_max_suppression_compose(pred_compose, iou_thres=args.nms_max_overlap)
        else:
            pred_compose_nms = pred_compose
            
        cont_det += len(pred_compose_nms[0])
            
        # Load classes names
        if args.det_model == "Detic":
            classes_names = model_predictor.metadata.thing_classes
        elif args.det_model == "grdSAM": 
            classes_names = []
            for ii in range(len(args.classes_grdSAM)):
                classes_names.append(args.classes_grdSAM[ii])
        else:
            classes_names = []
            for ii in range(len(model_predictor.names)):
                classes_names.append(model_predictor.names[ii])
        
        # Draw output img     
        img_out_bbox, img_out_mask, mask_tot = lists2img(np.asarray(img_o), pred_compose_nms, classes_names, fruit_zone)
        

        # Save results
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
    
    # Predict full image        
    else:
        start_time = time.time()
        empty_mask = np.zeros((img_p.shape[0], img_p.shape[1]))
        if args.det_model == "Detic":
            detections_p, img_out_bbox, img_out_mask, mask_tot, pred_str = detic_proccess_img(img_p, model_predictor, args, fruit_zone, empty_mask)
        elif args.det_model == "grdSAM": 
                    detections_p, img_out_bbox, img_out_mask, mask_tot, pred_str = grdSAM_process_image(img_p, model_predictor, args, fruit_zone, empty_mask)
        else:
            detections_p, img_out_bbox, img_out_mask, mask_tot, pred_str = yolo_proccess_img(img_p, model_predictor, args, fruit_zone, empty_mask)
                    
        cont_det += len(detections_p[0])
        logging.info("{}: {} in {:.2f}s".format(path, pred_str, time.time() - start_time)) 
        bboxs_p, confs_p, clss_p, masks_p, mask_p_agg = detections_p
        if isinstance(bboxs_p,torch.Tensor):
            pred_compose_nms = torch.FloatTensor(np.array(bboxs_p.cpu())), torch.FloatTensor(np.array(confs_p.cpu())), torch.FloatTensor(np.array(clss_p.cpu())), torch.FloatTensor(np.array(masks_p.cpu()))
        else: 
            pred_compose_nms = torch.FloatTensor(np.array(bboxs_p)), torch.FloatTensor(np.array(confs_p)), torch.FloatTensor(np.array(clss_p)), torch.FloatTensor(np.array(masks_p))


        # Save results
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
                
            cv2.imwrite(out_filename + "_vis.jpg", img_out_bbox)
            cv2.imwrite(out_filename + "_mask.jpg", img_out_mask)  
            
    return img_out_bbox, img_out_mask, mask_tot, cont_det, pred_compose_nms


def mask_to_coco_segmentation(mask, small_object_threshold=0.15):
    """Convert a binary segmentation mask to COCO 'segmentation' annotation format, excluding small blobs.

    Args:
        mask (ndarray): a 2D Numpy array of shape (H, W), where H is the height and W is the width of the image.
                        Each pixel is either 0 (background) or 1 (object).
        small_object_threshold (float): if the size of a blob is less than this fraction of the average blob size, it is removed.

    Returns:
        segmentation_data (dict): a dictionary containing the converted data in COCO format.
    """
    
    # Make sure the mask is in the correct format
    if not isinstance(mask, np.ndarray):
        raise ValueError("mask must be a 2D Numpy array")

    # Create a copy of the mask
    mask_copy = mask.copy()

    # Label different blobs in the mask
    labeled_mask, num_labels = ndimage.label(mask_copy)

    # Compute the size of each blob
    blob_sizes = ndimage.sum(mask_copy, labeled_mask, range(num_labels + 1))

    # Compute the average blob size
    average_blob_size = blob_sizes.mean()

    # Create a mask for small blobs
    small_blobs = np.isin(labeled_mask, np.where(blob_sizes < small_object_threshold * average_blob_size))

    # Remove small blobs
    mask_copy[small_blobs] = 0

    # Asegúrate de que la máscara esté en uint8
    mask_copy = mask_copy.astype(np.uint8)

    # Encuentra contornos en la máscara
    contours = measure.find_contours(mask_copy, 0.5)

    # Inicializa la lista de polígonos
    polygons = []

    # Inicializa el área total
    area = 0

    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)

        # Para manejar casos donde después de la simplificación nos quedamos con un polígono multiparte ("MultiPolygon")
        if poly.geom_type == 'MultiPolygon':
            # Unimos todos los polígonos en uno solo
            allparts = [p.buffer(0) for p in poly]
            poly = unary_union(allparts)

        if poly.geom_type == 'Polygon':
            x, y = poly.exterior.coords.xy
            poly_points = [(x[i], y[i]) for i in range(len(x))]
            polygons.append(poly_points)
            area += poly.area
            

    segmentation_data = {
        "segmentation": polygons 
    }

    return segmentation_data


def check_polygon_color(polygon, color_mask):
    """Check the color of the pixels within a polygon in a color segmentation mask.

    Args:
        polygon (list): a list of (x, y) pairs defining the vertices of the polygon.
        color_mask (ndarray): a 3D Numpy array of shape (H, W, 3), where H is the height and W is the width of the image,
                              and the third dimension represents the color channels (in RGB order).

    Returns:
        label (int): 0 if all non-black pixels within the polygon are blue in the color mask, 1 otherwise.
    """

    # Convert polygon coordinates to a 2D array (if not already)
    if isinstance(polygon[0], tuple):
        polygon = np.array(polygon)

    # The polygon coordinates should be in (row, col) format (y, x)
    polygon = np.fliplr(polygon)

    # Create a binary mask with the same shape as the color mask
    poly_mask = polygon2mask(color_mask.shape[:2], polygon)

    # Now, use this mask to get the pixels within the polygon from the color mask
    poly_pixels = color_mask[poly_mask]

    # Remove black pixels (consider them as background)
    non_black_pixels = poly_pixels[~np.all(poly_pixels == [0, 0, 0], axis=-1)]

    # Check if all these non-black pixels are blue
    if np.all(non_black_pixels == [255, 0, 0]):
        return 0
    else:
        return 1

def preprocess_img_trap (image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 2001, 1)
    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.dilate(thresh,kernel,iterations=2)
    return thresh

def detect_grid_contours(image):
    
    #edges = cv2.Canny(thresh,100,200, apertureSize=3)
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if (cv2.contourArea(cnt) > 1000 and cv2.contourArea(cnt) < 40000)]
    straight_contours = []
    for contour in contours:
        # Aproxima el contorno
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Si el contorno tiene cuatro puntos, es probablemente recto
        if len(approx) >= 4:
            straight_contours.append(contour)

    # Dibuja los contornos rectos en una copia de la imagen
    cv2.drawContours(image, straight_contours, -1, (0, 255, 0), 2)
    white_image = np.ones(image.shape, dtype=np.uint8) * 255
    cv2.drawContours(white_image, straight_contours, -1, (0, 0, 0), 3)

    return white_image

def calculate_angle(line):
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*a)
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*a)

    return np.arctan2(y2 - y1, x2 - x1)

def rho_distance(line1, line2):
    """Calcula la diferencia en rho entre dos líneas."""
    return abs(line1[0][0] - line2[0][0])

def filter_notortogonal_lines(lines, reference_angle, angle_threshold):
    filtered_lines = []
    for line in lines:
        # Calcula la diferencia de ángulos y envuelve alrededor de 2*pi
        angle_difference = abs(calculate_angle(line) - reference_angle) % (2 * np.pi)
        
        # Tomar la diferencia de ángulos en el intervalo [0, pi/2]
        angle_difference = min(angle_difference, 2*np.pi - angle_difference)
        
        # Comprueba si la línea es casi paralela o casi perpendicular a las líneas de referencia
        if angle_difference < angle_threshold or abs(angle_difference - np.pi/2) < angle_threshold:
            filtered_lines.append(line)
    return filtered_lines

def filter_close_lines(lines, rho_threshold=10):
    """
    Filtra las líneas que están muy cerca unas de otras.

    Las líneas deben estar en formato de Hough (rho, theta).

    El umbral para rho se puede ajustar si es necesario.
    """
    filtered_lines = []
    
    for line in lines:
        if not any(rho_distance(line, existing_line) < rho_threshold for existing_line in filtered_lines):
            filtered_lines.append(line)
            
    return filtered_lines

def average_rho_distance(lines, num_lines=10):
    """
    Calcula la distancia media entre las primeras 'num_lines' líneas en términos del parámetro rho.

    Las líneas deben estar en formato de Hough (rho, theta) y deben estar ordenadas por importancia/relevancia.
    """
    num_lines = min(num_lines, len(lines))
    if num_lines < 2:
        return(50)

    # Calcular todas las diferencias de rho entre pares de líneas
    rho_diffs = []
    for i in range(num_lines):
        for j in range(i+1, num_lines):
            rho_diff = abs(lines[i][0][0] - lines[j][0][0])
            rho_diffs.append(rho_diff)

    # Devolver la distancia media
    return np.mean(rho_diffs)

def filter_lines_by_angle(lines, min_angle=-30, max_angle=30):
    """
    Filtra las líneas que tienen un ángulo menor a 'min_angle'.

    Las líneas deben estar en formato de Hough (rho, theta).

    'min_angle' debe estar en grados.
    """
    min_angle_rad = np.radians(min_angle)  # Convertir el ángulo mínimo a radianes
    max_angle_rad = np.radians(max_angle)  # Convertir el ángulo mínimo a radianes
    filtered_lines = [line for line in lines if (line[0][1] >= min_angle_rad and line[0][1] <= max_angle_rad)]
    
    return filtered_lines

def print_lines(lines, shape, name="aux.jpg"):
    aux_image = np.ones(shape, dtype=np.uint8) * 255
    for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 2000*(-b))
            y1 = int(y0 + 2000*(a))
            x2 = int(x0 - 2000*(-b))
            y2 = int(y0 - 2000*(a))

            cv2.line(aux_image, (x1, y1), (x2, y2), (0, 0, 0), 2)
    cv2.imwrite(name, aux_image)


def detect_lines(image, orientation, thr=500):
    
    white_image = np.ones(image.shape, dtype=np.uint8) * 255
    
    lines = cv2.HoughLines(image, 1, np.pi/180, thr)

    # Calcula el ángulo promedio de las líneas de referencia
    reference_angle = orientation
    min_a_h = 90
    min_a_v = 0
    v_factor = 3
    h_factor = 6
    if reference_angle > (np.pi/2):
        reference_angle = reference_angle - np.pi/2
        min_a_h = 180
        min_a_v = 90
        v_factor = 6
        h_factor = 3

    # Establece un umbral para la diferencia de ángulo
    angle_threshold = np.pi / 120  # 2 grados

    # Filtra las líneas basadas en la diferencia de ángulo
    filtered_lines = filter_notortogonal_lines(lines, reference_angle, angle_threshold)

    vertical_filtered_lines = filter_lines_by_angle(filtered_lines, min_angle=min_a_v-30, max_angle=min_a_v+30)
    horizontal_filtered_lines = filter_lines_by_angle(filtered_lines, min_angle=min_a_h-30, max_angle=min_a_h+30)

    rho_threshold_vertical = average_rho_distance(vertical_filtered_lines, num_lines=500)
    rho_threshold_horizontal = average_rho_distance(horizontal_filtered_lines, num_lines=500)

    # Filter close lines
    filtered_lines_v = filter_close_lines(vertical_filtered_lines, rho_threshold_vertical/v_factor)
    filtered_lines_h = filter_close_lines(horizontal_filtered_lines, rho_threshold_horizontal/h_factor)

    filtered_lines = filtered_lines_v + filtered_lines_h
        
    for line in filtered_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 2000*(-b))
        y1 = int(y0 + 2000*(a))
        x2 = int(x0 - 2000*(-b))
        y2 = int(y0 - 2000*(a))

        cv2.line(white_image, (x1, y1), (x2, y2), (0, 0, 0), 2)

    return white_image,filtered_lines

def draw_overlapping_lines(lines1, lines2, img_shape):
    # Crea una imagen en blanco del mismo tamaño que la imagen original
    image = np.zeros(img_shape, dtype=np.uint8)

    # Para cada par de líneas
    lines_join = []
    for line1 in lines1:
        for line2 in lines2:
            #Verifica si las líneas se solapan
            rho_diff = abs(line1[0][0] - line2[0][0])
            theta_diff = abs(line1[0][1] - line2[0][1])

            rho_threshold = 500  # Ajusta estos umbrales según sea necesario
            theta_threshold = np.pi / 90  # Aproximadamente 1 grado

            if theta_diff < theta_threshold and rho_diff > rho_threshold:
                # Dibuja la línea en la imagen
                lines_join.append(line1)
    
    vertical_filtered_lines = filter_lines_by_angle(lines_join, min_angle=-30, max_angle=30)
    horizontal_filtered_lines = filter_lines_by_angle(lines_join, min_angle=60, max_angle=120)

    rho_threshold_vertical = average_rho_distance(vertical_filtered_lines, num_lines=50)
    rho_threshold_horizontal = average_rho_distance(horizontal_filtered_lines, num_lines=50)

    # Filter close lines
    filtered_lines_v = filter_close_lines(vertical_filtered_lines, rho_threshold_vertical/8)
    filtered_lines_h = filter_close_lines(horizontal_filtered_lines, rho_threshold_horizontal/8)

    filtered_lines = filtered_lines_v + filtered_lines_h

    for line in filtered_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 2000*(-b))
        y1 = int(y0 + 2000*(a))
        x2 = int(x0 - 2000*(-b))
        y2 = int(y0 - 2000*(a))

        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)


    return image

def find_intersections(image, lines):
    """
    Encuentra las intersecciones entre las líneas proporcionadas.

    Args:
        image: imagen original (sólo para pintar las intersecciones).
        lines: lista de líneas, donde cada línea es una tupla de (rho, theta).

    Returns:
        intersections: lista de puntos de intersección.
    """
    height, width = image.shape[:2]

    def line_intersection(line1, line2):
        rho1, theta1 = line1[0]
        rho2, theta2 = line2[0]
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        try:
            x0, y0 = np.linalg.solve(A, b)
            x0, y0 = int(np.round(x0)), int(np.round(y0))
            return [x0, y0] if 0 <= x0 < width and 0 <= y0 < height else None
        except np.linalg.LinAlgError:
            # Las líneas son paralelas (o son la misma línea), por lo que no se cruzan
            return None

    intersections = []
    for i, line1 in enumerate(lines):
        for line2 in lines[i+1:]:
            intersec = line_intersection(line1, line2)
            if intersec != None:
                x, y = intersec
                intersections.append((x, y))
                cv2.circle(image, (x, y), 10, (0, 255, 0), -1)

    return intersections, image

def draw_grid(image, grid, color=(0, 255, 0), thickness=2):
    """
    Dibuja una cuadrícula en una imagen.

    'image' es la imagen en la que dibujar la cuadrícula.
    'grid' es una lista de celdas de la cuadrícula, donde cada celda es una lista de las coordenadas de sus vértices.
    'color' es el color de las líneas de la cuadrícula.
    'thickness' es el grosor de las líneas de la cuadrícula.
    """
    img = image.copy()

    for cell in grid:
        pts = cell.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
        
    return img


def get_orientation(image):
    """
    Obtiene la orientación modal de las líneas verticales y horizontales en una imagen.

    'image' es la imagen de entrada.
    """
    #image_transposed = np.transpose(image, (1, 0))
    kernel= np.ones((20,20),np.uint8)
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    # Aplicar la transformada de Hough para líneas
    lines = cv2.HoughLines(gradient, 1, np.pi/180, 500)
    n=10
    lines = lines[:n]

    vertical_angles = []
    horizontal_angles = []

    for rho, theta in lines[:, 0]:
        # Convertir a grados
        theta_deg = np.degrees(theta)

        if 45 < theta_deg < 135:
            # Es una línea vertical
            vertical_angles.append(theta_deg)
        else:
            # Es una línea horizontal
            horizontal_angles.append(theta_deg)

    # Calcular la orientación modal
    vertical_orientation = mode(vertical_angles)[0][0] if vertical_angles else None
    horizontal_orientation = mode(horizontal_angles)[0][0] if horizontal_angles else None

    # Como la imagen entra en h,w en vez de w,h en realidad las h son las v
    orientation = horizontal_orientation

    return np.radians(orientation)


def draw_line(img, rho, theta):
    """
    Dibuja una línea en una imagen dada la orientación y el rho.

    'img' es la imagen de entrada,
    'rho' es la distancia desde el origen al punto más cercano en la línea,
    'theta' es el ángulo formado por esta línea perpendicular y la horizontal.
    """
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)


def detect_grid(args, img_zoi_crop, output_folder="", file_name=""):
    # Preprocess image
    img_prep = preprocess_img_trap(np.copy(img_zoi_crop))
    if args.debug:
        cv2.imwrite(os.path.join(output_folder,"preprocess_" + file_name), img_prep)
    # Detect the grid
    img_grid = detect_grid_contours(np.copy(img_prep))
    if args.debug:
        cv2.imwrite(os.path.join(output_folder,"grid_" + file_name), img_grid)

    orientation = get_orientation(img_grid)

    # Detect the lines
    img_lines_raw, lines_raw = detect_lines(np.copy(img_prep), orientation, thr=300)
   
    if args.debug:
        cv2.imwrite(os.path.join(output_folder,"linesraw_" + file_name), img_lines_raw)

    grid_contours, img_grid_contour =  draw_and_number_square_contours(np.copy(img_lines_raw))

    if args.debug:
        cv2.imwrite(os.path.join(output_folder,"grid_cells_" + file_name), img_grid_contour)
    
    return grid_contours


class Cell:
    def __init__(self, top_left, top_right, bottom_right, bottom_left):
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_right = bottom_right
        self.bottom_left = bottom_left


def create_cells(grid_points, width, height):
    cells = []
    num_cells = len(grid_points) - 1

    for i in range(num_cells):
        for j in range(num_cells):
            top_left = grid_points[i][j]
            top_right = grid_points[i][j + 1]
            bottom_right = grid_points[i + 1][j + 1]
            bottom_left = grid_points[i + 1][j]

            # Verificar si todos los vértices de la celda están dentro de los límites de la imagen
            if (0 <= top_left[0] < width and 0 <= top_left[1] < height and
                0 <= top_right[0] < width and 0 <= top_right[1] < height and
                0 <= bottom_right[0] < width and 0 <= bottom_right[1] < height and
                0 <= bottom_left[0] < width and 0 <= bottom_left[1] < height):

                cell = Cell(top_left, top_right, bottom_right, bottom_left)
                cells.append(cell)



import cv2
import numpy as np

def draw_cells(width, height, cells):
    # Crear una imagen en blanco
    image = np.zeros((height, width, 3), np.uint8)

    # Generar un color único para cada celda
    for cell in cells:
        color = tuple(np.random.randint(0, 255, 3).tolist())

        # Dibujar la celda en la imagen
        pts = np.array([cell.top_left, cell.top_right, cell.bottom_right, cell.bottom_left], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(image, [pts], color)

    return image


def get_detection_statistics(image, roi, detections, grid_points):
    # Desempacar las coordenadas de la zona de interés
    x_min, y_min, x_max, y_max = roi
    
    # Establecer una matriz para almacenar el número de detecciones en cada celda
    num_cells = len(grid_points) - 1
    detections_per_cell = np.zeros((num_cells, num_cells))

    # Calcular el ancho y alto de las celdas
    cell_width = (x_max - x_min) / num_cells
    cell_height = (y_max - y_min) / num_cells
    
    # Iterar sobre las detecciones
    for x, y in detections:
        # Solo considerar las detecciones dentro de la zona de interés
        if x_min <= x < x_max and y_min <= y < y_max:
            # Determinar en qué celda se encuentra la detección
            cell_x = int((x - x_min) / cell_width)
            cell_y = int((y - y_min) / cell_height)

            # Incrementar el recuento de detecciones para esa celda
            detections_per_cell[cell_y, cell_x] += 1

    return detections_per_cell


def draw_and_number_square_contours(img_bin, tolerance=0.2, filter=False):
    # Llamar a la función anterior para encontrar los contornos cuadrados
    square_contours = find_square_contours(img_bin)

    # Calcular el área de cada contorno
    areas = [cv2.contourArea(cnt) for cnt in square_contours]
    
    # Calcular el valor modal de las áreas
    mean_area = np.mean(areas)

    if filter:
        # Filtrar los contornos cuadrados basándose en su área
        filtered_contours = [cnt for cnt, area in zip(square_contours, areas) if  mean_area <= area]
    else:
        filtered_contours = square_contours

    # Convertir la imagen binaria a color para poder dibujar contornos en color
    img_color = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)

    # Dibujar cada contorno cuadrado en la imagen, con un número de identificación
    for i, cnt in enumerate(filtered_contours):
        cv2.drawContours(img_color, [cnt], -1, (0, 255, 0), 2)

        # Calcular el centroide del contorno
        M = cv2.moments(cnt)
        if M["m00"] != 0: # evita dividir por cero
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # Dibujar el número de identificación en el centroide
        cv2.putText(img_color, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Devolver la imagen con los contornos dibujados y numerados
    return filtered_contours, img_color

def find_square_contours(img_bin):
    # Encontrar contornos en la imagen
    contours, _ = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar solo los contornos cerrados y cuadrados
    square_contours = []
    for cnt in contours:
        # Calcular el área del contorno
        area = cv2.contourArea(cnt)

        # Si el área es 0, el contorno no es cerrado
        if area > 0:
            # Calcular el perímetro del contorno
            perimeter = cv2.arcLength(cnt, True)

            # Calcular la relación entre el área del contorno y el área de un cuadrado con el mismo perímetro
            squareness = 4 * np.pi * area / (perimeter ** 2)

            # Si la relación es cercana a 1, el contorno es cuadrado
            if 0.65 < squareness < 1.35:
                square_contours.append(cnt)

    return square_contours


def insect_statistics(predictions, roi_bbox, grid_cells, clss_names, area_cell_std=625):
    if grid_cells != None:
        # Transform grid cell coordinates to full image coordinates
        grid_cells_abs = [np.squeeze(cell, axis=1) + (roi_bbox[1], roi_bbox[0]) for cell in grid_cells]

        # Calculate the area of each grid cell
        cell_areas = [cv2.contourArea(cell) for cell in grid_cells_abs]

        # Calculate the mean area of the grid cells
        mean_cell_area = np.mean(cell_areas)

        # Sort cell areas in descending order
        sorted_cell_areas = sorted(cell_areas, reverse=True)
        
        # Select the top 10 cell areas
        top_20_cell_areas = sorted_cell_areas[:20]

        # Detect and remove outliers using IQR
        q1 = np.percentile(top_20_cell_areas, 25)
        q3 = np.percentile(top_20_cell_areas, 75)
        iqr = q3 - q1
        lower_bound = q1 - (1.2 * iqr)
        upper_bound = q3 + (1.2 * iqr)
        filtered_cell_areas = [area for area in top_20_cell_areas if lower_bound <= area <= upper_bound]

        # Calculate the mean of the filtered cell areas
        mean_filtered_area = np.mean(filtered_cell_areas)

        # Initialize dictionary to store statistics for each cell
        cell_stats = {i: {"coords": cell.tolist(), "area_mm2": round((area_cell_std * area)/(mean_filtered_area),2)} for i, (cell, area) in enumerate(zip(grid_cells_abs, cell_areas))}

        total_detections = 0
        bboxes, confs, clss, masks = predictions

        # For each detection...
        for ii in range(len(bboxes)):
            # Get the coordinates of the center of the detection
            x_center = int((bboxes[ii][0] + bboxes[ii][2]) / 2)
            y_center = int((bboxes[ii][1] + bboxes[ii][3]) / 2)

            # Check in which cell the center of the detection falls
            for i, cell in enumerate(grid_cells_abs):
                if cv2.pointPolygonTest(cell.reshape(-1,1,2), (x_center, y_center), False) >= 0:
                    # If the class of the detection has already been detected in this cell, increment the count
                    if clss_names[int(clss[ii])] in cell_stats[i]:
                        cell_stats[i][clss_names[int(clss[ii])]] += 1
                    # If not, initialize the count to 1
                    else:
                        cell_stats[i][clss_names[int(clss[ii])]] = 1
                    total_detections += 1

        # Calculate the total density of detections per cell (in insects per mm^2)
        total_density_per_cell = round(total_detections / sum(cell_areas) * 25.0**2, 3)

        # Calculate the density of each class of insects per cell (in insects per mm^2)
        class_counts = {cls: sum([cell_stats[i].get(cls, 0) for i in range(len(grid_cells_abs))]) for cls in clss_names}
        class_density_per_cell = {cls: round(count / sum(cell_areas) * 25.0**2, 3) for cls, count in class_counts.items()}

        # Create the new dictionary that includes the "grid" structure
        output_stats = {
            "grid": cell_stats,
            "mean_cell_area": round((area_cell_std * mean_cell_area)/(mean_filtered_area),2),  # Convert to mm^2
            "total_density_per_cell": total_density_per_cell,
            "class_density_per_cell": class_density_per_cell,
            "total_detections": total_detections
        }
    else:
        total_detections = 0
        bboxes, confs, clss, masks = predictions
        class_stats = {}

        # For each detection...
        for ii in range(len(bboxes)):
            if clss_names[int(clss[ii])] in class_stats:
                class_stats[clss_names[int(clss[ii])]] += 1
            # If not, initialize the count to 1
            else:
                class_stats[clss_names[int(clss[ii])]] = 1
        
            total_detections += 1

        # Calculate the density of each class of insects per cell (in insects per mm^2)
        output_stats = {
            "grid": "No grid detected",
            "class_density_per_cell": class_stats,
            "total_detections": total_detections
        }


    return output_stats


def list_image_paths(path):
    # Crear una lista para almacenar las rutas de las imágenes
    image_paths = []

    # Comprobar si la ruta es a un directorio
    if os.path.isdir(path):
        # Recorrer todos los archivos en el directorio
        for filename in os.listdir(path):
            # Comprobar si el archivo es una imagen
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                # Si es una imagen, añadir su ruta a la lista
                image_paths.append(os.path.join(path, filename))
    # Comprobar si la ruta es a un archivo
    elif os.path.isfile(path) and path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        # Si es una imagen, añadir su ruta a la lista
        image_paths.append(path)

    return image_paths


