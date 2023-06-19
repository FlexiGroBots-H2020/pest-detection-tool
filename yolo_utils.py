import numpy as np
import cv2
import random
import torch

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


def patch_yolo2list(empty_mask, yolo_pred, patch_size, row, column, overlap, fruit_zone):
    bboxes, confs, clss, masks = yolo_pred 
    bboxes = bboxes.cpu().numpy()
    confs = confs.cpu().numpy()
    clss = clss.cpu().numpy()
    
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

def yolo_rel2abs(orig_img, process_img, bboxes, masks):
    # Get dimensions of original and process images
    orig_h, orig_w, _ = orig_img.shape
    proc_h, proc_w, _ = process_img.shape
    
    # Calculate scale factors for translation
    scale_x = orig_w / proc_w
    scale_y = orig_h / proc_h
    
    # Translate bounding boxes to original image dimensions
    translated_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox.tolist()
        translated_bbox = [int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)]
        translated_bboxes.append(translated_bbox)
    
    # Translate masks to original image dimensions
    translated_masks = []
    if masks is not None:
        for mask in masks:
            # Convert mask tensor to NumPy array
            mask = mask.squeeze().cpu().numpy()
            # Resize mask to process image dimensions
            resized_mask = cv2.resize(mask, (proc_w, proc_h))
            # Translate mask to original image dimensions
            translated_mask = cv2.resize(resized_mask, (orig_w, orig_h))
            # Convert mask back to PyTorch tensor
            #translated_mask = torch.from_numpy(translated_mask)
            translated_masks.append(translated_mask)
    
    return torch.Tensor(translated_bboxes), torch.Tensor(translated_masks)

def yolo_proccess_img(img, yolo_predictor, args, fruit_zone, empty_mask, patch_id = (0,0)):
    # Predict
    if img.shape != (640,640,3):
        img_yolo = cv2.resize(img, (640,640)) # fix image to model input dimensions
    else:
        img_yolo = img
        
    results = yolo_predictor(img_yolo, imgsz=640, conf=args.confidence_threshold, iou=args.nms_max_overlap) 
    
    # batch size of input = 1
    detections = results[0].boxes
    
    pred_str = "detected {} instances".format(len(detections))
    
    if len(detections) > 0: 
        bboxs_p = detections.xyxy
        confs_p = detections.conf
        class_p = detections.cls
        if results[0].masks!=None:
            masks_p = results[0].masks.data
        else:
            masks_p = None
        
        # Translate detections to original dimensions
        if img_yolo.shape != img.shape:  
            bboxs_p, masks_p = yolo_rel2abs(img, img_yolo, bboxs_p, masks_p)
        
        predictions_nms = bboxs_p, confs_p, class_p, masks_p
        
        # Take classes names
        classes_names = []
        for ii in range(len(yolo_predictor.names)):
            classes_names.append(yolo_predictor.names[ii])
            
        # Draw detections over image and save it 
        img_out_bbox = draw_detections(img, predictions_nms, classes_names)
        
        # Generate aggregation masks
        img_out_mask, mask_tot = generate_final_mask(masks_p, img)
        
        # Save detections in full img coordinates
        if args.patching:
            bboxs, confs, clss, masks, mask_agg = patch_yolo2list(empty_mask, predictions_nms, args.patch_size, patch_id[0], patch_id[1], args.overlap, fruit_zone)
        else:
            bboxs, confs, clss, masks, mask_agg = bboxs_p, confs_p, class_p, masks_p, mask_tot
    else:
        img_out_bbox = img
        img_out_mask = np.zeros((img.shape[0], img.shape[1], 3))
        mask_tot = np.zeros((img.shape[0], img.shape[1]))
        bboxs, confs, clss, masks, mask_agg = np.empty(0), np.empty(0), np.empty(0), empty_mask, empty_mask
    
    return ((bboxs, confs, clss, masks, mask_agg), img_out_bbox, img_out_mask, mask_tot, pred_str)