import cv2
import numpy as np
import supervision as sv

import torch
import torchvision


from GroundedSAM.GroundingDINO.groundingdino.util.inference import Model
from GroundedSAM.samhq.segment_anything import sam_model_registry, SamPredictor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import argparse

def get_parser_grdSAM():
    """
    Creates and returns an argument parser for input configuration.
    """
    parser = argparse.ArgumentParser(description="Argument parser for image processing configurations")
    
    # Classes for object detection
    parser.add_argument(
        "--classes_grdSAM",
        default=[],
        nargs='+',
        help="List of classes to detect."
    )

    parser.add_argument(
        "--classes_grdSAM_hl",
        default=[],
        nargs='+',
        help="List of classes to highlight with background extraction."
    )

    # Thresholds
    parser.add_argument(
        "--box_threshold_grdSAM",
        type=float,
        default=0.25,
        help="Box threshold for GroundingDINO."
    )
    parser.add_argument(
        "--text_threshold_grdSAM",
        type=float,
        default=0.25,
        help="Text threshold for GroundingDINO."
    )
    parser.add_argument(
        "--nms_threshold_grdSAM",
        type=float,
        default=0.8,
        help="Non-maximum suppression threshold."
    )

    parser.add_argument(
        "--save_imgs",
        type=bool,
        default=True,
        help="Save resulting images"
    )

    return parser.parse_known_args()


def load_model_grdSAM():
    """
    Loads GroundingDINO and SAM models.
    
    Returns:
    - grounding_dino_model: GroundingDINO inference model.
    - sam: SAM model.
    - sam_predictor: SAM model predictor.
    """
    # GroundingDINO config and checkpoint paths
    GROUNDING_DINO_CONFIG_PATH = "./GroundedSAM/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "./models/groundingdino_swinb_cogcoor.pth"

    # Segment-Anything checkpoint configuration
    SAM_ENCODER_VERSION = "vit_l"
    SAM_CHECKPOINT_PATH = "./models/sam_hq_vit_l.pth"

    # Building the GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    return grounding_dino_model, sam_predictor


def grdSAM_process_image(image, model_predictor, args, fruit_zone, empty_mask, patch_id = (0,0)):
    """
    Processes an image using the GroundingDINO and SAM models for object detection and segmentation.
    
    Args:
    - image (np.ndarray): The input image.
    - classes (list): List of classes for detection.
    - classes_hl (list): List of classes for highlight.
    - box_threshold (float): Box threshold for GroundingDINO.
    - text_threshold (float): Text threshold for GroundingDINO.
    - nms_threshold (float): Non-maximum suppression threshold.
    
    Returns:
    - annotated_frame (np.ndarray): Image annotated with GroundingDINO detections.
    - annotated_image (np.ndarray): Image annotated with GroundedSAM segmentations.
    - output_image (np.ndarray): Image highlighting objects of class 0.
    """
    
    # Detect objects in the image using the GroundingDINO model
    grounding_dino_model, sam_predictor = model_predictor
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=args.classes_grdSAM,
        box_threshold=args.box_threshold_grdSAM,
        text_threshold=args.text_threshold_grdSAM
    )
    if len(detections) > 0:
        # Annotate the image with the detected objects
        box_annotator = sv.BoxAnnotator()
        labels = [f"{args.classes_grdSAM[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections]
        img_out_bbox = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

        # Apply non-maximum suppression to reduce overlapping detections
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            args.nms_threshold_grdSAM
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        # Convert the detections to segmentation masks using the SAM model
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        # Filter out detections that are not of class hl
        list_clsidx_hl = []
        if args.classes_grdSAM_hl == []:
            args.classes_grdSAM_hl = args.classes_grdSAM # All detections HL
    
        for ii in range(len(args.classes_grdSAM)):
            if args.classes_grdSAM[ii] in args.classes_grdSAM_hl:
                list_clsidx_hl.append(ii)
        hl_array =  np.array(list_clsidx_hl) 
        class_hl_detections = detections.empty()
        list_det = []
        for ii in range(len(detections)):
            if detections[ii].class_id[0] in hl_array:
                list_det.append(detections[ii])
        class_hl_detections = class_hl_detections.merge(list_det)

        # Annotate the image with the segmentation masks
        mask_annotator = sv.MaskAnnotator()
        img_out_mask = mask_annotator.annotate(scene=image.copy(), detections=class_hl_detections)
        img_out_mask = box_annotator.annotate(scene=img_out_mask, detections=class_hl_detections, labels=labels)


        # Create an output image with 4 channels (RGB + Alpha)
        output_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=image.dtype)

        class_hl_mask = np.zeros((image.shape[0], image.shape[1]))

        if class_hl_detections:
            # Merge all masks of class 0 objects
            chl_masks = [mask for mask in [detection[1] for detection in class_hl_detections]]
            for ii in range(len(chl_masks)):
                class_hl_mask += chl_masks[ii]

            # Highlight the class 0 objects in the output image
            # Fill the RGB channels
            output_image[class_hl_mask >= 1, :3] = image[class_hl_mask >= 1]

            # Set the alpha channel to 255 (opaque) for the objects of interest and 0 (transparent) for the background
            output_image[class_hl_mask >= 1, 3] = 255

        bboxs_p, confs_p, class_p, masks_p = class_hl_detections.xyxy, class_hl_detections.confidence, class_hl_detections.class_id, class_hl_detections.mask 
        predictions_nms = bboxs_p, confs_p, class_p, masks_p
        # Save detections in full img coordinates
        if args.patching:
            if len(class_hl_detections) > 0:
                bboxs, confs, clss, masks, mask_agg = patch_grdSAM2list(empty_mask, predictions_nms, args.patch_size, patch_id[0], patch_id[1], args.overlap, fruit_zone)
            else:
                bboxs, confs, clss, masks, mask_agg = bboxs_p, confs_p, class_p, masks_p, empty_mask
        else:
            bboxs, confs, clss, masks, mask_agg = bboxs_p, confs_p, class_p, masks_p, masks_p
    else:
        img_out_bbox = image
        img_out_mask = np.zeros((image.shape[0], image.shape[1], 3))
        mask_tot = np.zeros((image.shape[0], image.shape[1]))
        output_image = mask_tot
        labels = ""
        bboxs, confs, clss, masks, mask_agg = np.empty(0), np.empty(0), np.empty(0), empty_mask, empty_mask
    
    return ((bboxs, confs, clss, masks, mask_agg), img_out_bbox, img_out_mask, output_image, labels)


def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """
    Generates segmentation masks using the SAM model.
    
    Args:
    - sam_predictor (SamPredictor): SAM model predictor.
    - image (np.ndarray): Input image.
    - xyxy (np.ndarray): Bounding box coordinates.
    
    Returns:
    - np.ndarray: Segmentation masks.
    """
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

# Example usage:
# grounding_dino_model, sam, sam_predictor = load_models()
# image, classes = parse_inputs("./assets/demo2.jpg", ["The running dog"])
# grounded_dino_result, grounded_sam_result = process_image(image, classes)
# cv2.imwrite("groundingdino_annotated_image.jpg", grounded_dino_result)
# cv2.imwrite("grounded_sam_annotated_image.jpg", grounded_sam_result)

def patch_grdSAM2list(empty_mask, yolo_pred, patch_size, row, column, overlap, fruit_zone):
    bboxes, confs, clss, masks = yolo_pred 
    
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
        total_mask_patch = total_mask_patch + masks[jj]
        if total_mask_patch.shape != window_shape:
                total_mask_patch = total_mask_patch[0:window_shape[0], 0:window_shape[1]]
        total_mask[round(row*patch_size*(1 - overlap)):round(row*patch_size*(1 - overlap) + patch_size), round(column*patch_size*(1 - overlap)):round(column*patch_size*(1 - overlap) + patch_size)] = total_mask_patch
    if np.max(total_mask)>1:
        total_mask[total_mask > 1] = 1

    return bboxes, confs, clss, masks, total_mask