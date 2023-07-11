import multiprocessing as mp

from pest_detection_utils import list_image_paths, get_parser, load_detic, load_xdecoder_semseg, process_mask, mask_metrics, predict_img, pred2COCOannotations, detect_grid, insect_statistics

import sys
sys.path.insert(0, 'detectron2/')
sys.path.insert(0, 'Detic/')
sys.path.insert(0, 'Detic/third_party/CenterNet2/')
sys.path.insert(0, 'X_Decoder/')

from detectron2.utils.logger import setup_logger
from xdcoder_utils import semseg_single_im
from PIL import Image
import cv2
import os
import numpy as np
import glob
import json
import time
import logging

from ultralytics import YOLO

# constants
WINDOW_NAME = "Detic"

if __name__ == "__main__":
    
    gt_data = False
    
    logging.basicConfig(level=logging.INFO)
    
    # Set-up models and variables
    setup_logger(name="fvcore")
    logger = setup_logger()
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    
    logger.info("Arguments: " + str(args))
    
    if args.det_model == "Detic":
        model_predictor = load_detic(args)
    else:
        model_predictor = YOLO("yolov8l.pt")  # load an official model
        model_predictor = YOLO("models/best_46_epoch_pest.pt")  # load a custom model
    
    if args.full_pipeline:
        #model, transform, metadata, vocabulary_xdec = load_xdecoder_refseg(args)
        model, transform, metadata, vocabulary_xdec = load_xdecoder_semseg(args)
    
    list_images_paths = list_image_paths(args.input[0])     
            
    # Generate experiment folder 
    list_existing_exp = glob.glob(os.path.join(args.output, "exp*"))
    exist_exp_idx = np.zeros(len(list_existing_exp),dtype=int)
    for ii in range(len(list_existing_exp)):
        exist_exp_idx[ii] = int(list_existing_exp[ii].split("exp")[1])
    for jj in range(len(list_existing_exp)+1):
        if jj not in exist_exp_idx:
            exp_name= "exp" + str(jj)
    exp_folder = os.path.join(args.output, exp_name)
    if not os.path.exists(exp_folder):
            os.makedirs(exp_folder)
    
    # Generate experiment json file
    variables = {
        'path': exp_folder,
        'patching': args.patching,
        'patch_size': args.patch_size,
        'overlap': args.overlap, 
        'vocabulary_xdec': args.vocabulary_xdec,
        'vocabulary_detic': args.custom_vocabulary,
        'conf_threshold': args.confidence_threshold,
        'detection_model' : args.det_model, 
        'full_pipeline' : args.full_pipeline,
        'segmentation_step': args.seg_step
    }
    json_path = os.path.join(exp_folder,'variables.json')
    with open(json_path, 'w') as f:
        f.write(json.dumps(variables))
        
    # Load classification model: clss 0 (Botrytis/Damaged) and clss 1 (Healthy)
    #yolo_clss_health = YOLO("yolov8l-cls.pt")  # load an official model
    #yolo_clss_health = YOLO("models/best_health_cls_v2.pt")  # load a custom model
    
    metrics = []    
    cont_healthy = 0
    cont_unhealthy = 0
    cont_det = 0
    # Process each input image
    for img_path in list_images_paths:
        start_time = time.time()
        # Set the paths to the images/outputs and GT data
        gt_path = os.path.join(img_path.split("/images")[0], "instances")
        file_name = img_path.split('/')[-1]
        base_name = file_name.split('.')[-2]
        if os.path.exists(gt_path): 
            row_id = img_path.split("/")[-4] # keep dataq format
        else: 
            row_id = "img"
          
        output_folder = os.path.join(exp_folder, row_id + "_" + base_name) 
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
        # Load img in PILL and CV2
        img = Image.open(img_path).convert("RGB")
        #img = Image.fromarray(np.asarray(img)[..., ::-1])
        img_ori_np = cv2.cvtColor(np.asarray(img),cv2.COLOR_BGR2RGB)
        
        # Load GT data if exists
        if os.path.exists(gt_path):
            gt_mask = cv2.imread(os.path.join(gt_path,file_name),cv2.IMREAD_GRAYSCALE)
            if args.debug:
                cv2.imwrite(os.path.join(output_folder,"gtMask_" + file_name), gt_mask*255)
            gt_data = True
        else:
            gt_data = False
        
        if args.full_pipeline:
            
            if args.seg_step:
                seg_start_time = time.time()
                try:
                    logger.info("first pixel color check bf seg: {}".format(np.asarray(img)[0][0]))
                    #img_zoi_crop, fruit_bbox, img_seg = refseg_single_im(img, vocabulary_xdec, transform, model, metadata, output_folder, base_name, save=False)
                    img_zoi_crop, fruit_bbox, img_seg = semseg_single_im(img, transform, model, metadata, output_root="", file_name = "sem.png", save=False)
                except:
                    out_img = cv2.cvtColor(np.asarray(img_ori_np), cv2.COLOR_BGR2RGB)
                    fruit_zone = (0,0,out_img.shape[0],out_img.shape[1]) # top left down right
                    fruit_bbox = fruit_zone
                    img_seg= cv2.cvtColor(np.asarray(img_ori_np), cv2.COLOR_BGR2RGB)
                    img_zoi_crop = cv2.cvtColor(np.asarray(img_ori_np), cv2.COLOR_BGR2RGB)
                    logger.info("error segmentating")
                if args.debug:
                    cv2.imwrite(os.path.join(output_folder,"crop_" + file_name), img_zoi_crop)
                    cv2.imwrite(os.path.join(output_folder,"seg_" + file_name), img_seg)
                logger.info("Segmentation time: {:.2f} seconds".format(time.time() - seg_start_time))
            else:
                out_img = cv2.cvtColor(np.asarray(img_ori_np), cv2.COLOR_BGR2RGB)
                fruit_zone = (0,0,out_img.shape[0],out_img.shape[1]) # top left down right
                fruit_bbox = fruit_zone
                img_seg= cv2.cvtColor(np.asarray(img_ori_np), cv2.COLOR_BGR2RGB)
                img_zoi_crop = cv2.cvtColor(np.asarray(img_ori_np), cv2.COLOR_BGR2RGB)
                
            grid_start_time = time.time()
            try:
                grids = detect_grid(args, img_zoi_crop, output_folder=output_folder, file_name=file_name)
            except Exception as e:
                grids = None
                logger.info("No grid detected: {}".format(e))
            logger.info("Grid detection time: {:.2f} seconds".format(time.time() - grid_start_time))    

            det_start_time = time.time()
            # Predict with detection model over patches or full image
            logger.info("first pixel color check bf det crop: {}".format(img_zoi_crop[0][0]))
            logger.info("first pixel color check bf det img_ori: {}".format(img_ori_np[0][0]))
            img_out_bbox, img_out_mask, mask, img_health, health_flag, cont_det, pred = predict_img(img_zoi_crop, args, model_predictor, save=True, save_path=output_folder, img_o=img_ori_np, fruit_zone=fruit_bbox, health_model=None, cont_det=cont_det)
            logger.info("Detection time: {:.2f} seconds".format(time.time() - det_start_time))
            
            post_start_time = time.time()
            try:
                if args.det_model == "Detic":
                    clss_names = model_predictor.metadata.thing_classes
                else:
                    clss_names = model_predictor.names
                pest_stats = insect_statistics(pred, fruit_bbox, grids, clss_names)
            except Exception as e:
                logger.info("No grid statistics obtained: {}".format(e))
                pest_stats = ""
                
            # Generate final mask --> post-process
            if args.det_model == "Detic":
                mask_final = process_mask(mask, save=False, save_path=output_folder, max_clusters=30)
            else:
                mask_final = mask
                
            logger.info("Post-processing time: {:.2f} seconds".format(time.time() - post_start_time))
            
            
            # Compare results with GT data if exists
            if gt_data and cv2.countNonZero(mask_final):
                iou, dice, jaccard, hausdorff = mask_metrics(gt_mask, mask_final)
                logger.info("IoU:" + str(iou) + " F1-Dice:" + str(dice) + " Jacc:" + str(jaccard) + " Haus:" + str(hausdorff))
                metrics.append([{"iou":iou}, {"Dice": dice}, {"Jaccard": jaccard}, {"Haus": hausdorff}, {"Time": (time.time()-start_time)}])
            
            # Save output images
            if args.debug:
                cv2.imwrite(os.path.join(output_folder,"final_" + file_name), img_out_bbox)
                cv2.imwrite(os.path.join(output_folder,"finalMask_" + file_name), img_out_mask)
                if args.print_health:
                     cv2.imwrite(os.path.join(output_folder,"health_" + file_name), img_health)
                if args.det_model == "Detic":
                    out_img_masked = cv2.bitwise_and(img_ori_np, img_ori_np,  mask=mask_final.astype("uint8"))
                    cv2.imwrite(os.path.join(output_folder,"finalMaskProccessed_" + file_name), out_img_masked)
                
                # EXTRA CODE TO SAVE MASKS AS DATASET
                output_folder.split("/")[:-1][0]
                extra_folder = os.path.join(output_folder.split("/")[:-1][0], output_folder.split("/")[:-1][1], "mask_data")
                if not os.path.exists(extra_folder):
                    os.makedirs(extra_folder)
                if np.max(img_out_mask) > 0:
                    cv2.imwrite(os.path.join(extra_folder, file_name), img_out_mask)
                    
                # Generate metrics json file
                json_path = os.path.join(extra_folder,'stats.json')
                with open(json_path, 'w+') as f:
                    f.write("Healthy {}/{} and Unhealthy{}/{}\n".format(cont_healthy,cont_healthy+cont_unhealthy,cont_unhealthy,cont_healthy+cont_unhealthy))
                    f.write("Num detections {}\n".format(cont_det))
                    f.write("Time: {}\n".format((time.time()- seg_start_time)))
                
        else:
            # Predict with detection model over patches or full image
            det_start_time = time.time()
            img_out_bbox, img_out_mask, mask, img_health, health_flag, cont_det, pred = predict_img(img_ori_np, args, model_predictor, save=False, save_path=output_folder, health_model=None, cont_det=cont_det)
            logger.info("Detection time: {:.2f} seconds".format(time.time() - det_start_time))
            
            post_start_time = time.time()
            if health_flag == True:
                cont_unhealthy+=1
            else:
                cont_healthy+=1
            

            # Generate final mask --> post-process
            if args.det_model == "Detic":
                mask_final = process_mask(mask, save=False, save_path=output_folder, max_clusters=30)
            else:
                mask_final = mask
            logger.info("Post-processing time: {:.2f} seconds".format(time.time() - post_start_time))
            
            # Compare results with GT data if exists
            if gt_data and cv2.countNonZero(mask_final):
                iou, dice, jaccard, hausdorff = mask_metrics(gt_mask, mask_final)
                logger.info("IoU:" + str(iou) + " F1-Dice:" + str(dice) + " Jacc:" + str(jaccard) + " Haus:" + str(hausdorff))
                metrics.append([{"iou":iou}, {"Dice": dice}, {"Jaccard": jaccard}, {"Haus": hausdorff}])
            
            # Save output images
            if args.debug:
                cv2.imwrite(os.path.join(output_folder,"final_" + file_name), img_out_bbox)
                cv2.imwrite(os.path.join(output_folder,"finalMask_" + file_name), img_out_mask)
                if args.det_model == "Detic":
                    out_img_masked = cv2.bitwise_and(img_ori_np, img_ori_np,  mask=mask_final.astype("uint8"))
                    cv2.imwrite(os.path.join(output_folder,"finalMaskProccessed_" + file_name), out_img_masked)
                if args.print_health:
                    cv2.imwrite(os.path.join(output_folder,"health_" + file_name), img_health)
                    
                # EXTRA CODE TO SAVE MASKS AS DATASET
                output_folder.split("/")[:-1][0]
                extra_folder = os.path.join(output_folder.split("/")[:-1][0], output_folder.split("/")[:-1][1], "mask_data")
                if not os.path.exists(extra_folder):
                    os.makedirs(extra_folder)
                cv2.imwrite(os.path.join(extra_folder, file_name), out_img_masked)
                # Generate metrics json file
                json_path = os.path.join(extra_folder,'stats.json')
                with open(json_path, 'w+') as f:
                    f.write("Healthy {}/{} and Unhealthy{}/{}\n".format(cont_healthy,cont_healthy+cont_unhealthy,cont_unhealthy,cont_healthy+cont_unhealthy))
                    f.write("Num detections {}\n".format(cont_det))
                    f.write("Time: {}\n".format((time.time()-det_start_time)))
                    
        logger.info("Total processing time for {}: {:.2f} seconds".format(file_name, time.time() - start_time))
                
        # Save preds as annotations.txt
        txt_path = os.path.join(exp_folder,"annotations") 
        if not os.path.exists(txt_path):
            os.mkdir(txt_path)
            
        logging.info("img_ori_np shape, color: {}, {}".format(img_ori_np.shape, img_ori_np[0][0]))
        logging.info("mask_final shape, color: {}, {}".format(mask_final.shape, mask_final[0][0]))
        #logging.info("img_health shape, color: {}, {}".format(img_health.shape, img_health[0][0]))
        annotations = pred2COCOannotations(img_ori_np, mask_final, img_health, pred)

        txt_file = os.path.join(txt_path,"annotations.txt") 
        with open(json_path, 'w') as f:
            f.write(json.dumps(annotations))    
        # Generate metrics json file
        json_path = os.path.join(exp_folder,'pest.json')
        with open(json_path, 'w') as f:
            f.write(json.dumps(pest_stats))

    # Obtain mean metrics and save if GT data exists
    if gt_data: 
        # Calculate and save mean metrics values
        m_iou = 0
        m_f1 = 0 
        m_jc = 0
        m_hf = 0
        m_t = 0
        for ii in range(len(metrics)):
            m_iou = m_iou + metrics[ii][0]["iou"]
            m_f1 = m_f1 + metrics[ii][1]["Dice"]
            m_jc = m_jc + metrics[ii][2]["Jaccard"]
            m_hf = m_hf + metrics[ii][3]["Haus"]
            m_t = m_t + metrics[ii][4]["Time"]
            
        m_iou = m_iou / (ii+1)
        m_f1 = m_f1 / (ii+1)
        m_jc = m_jc / (ii+1)
        m_hf = m_hf / (ii+1)
        m_t = m_t / (ii+1)
        
        logger.info("mean IoU:" + str(m_iou) + " mean F1-Dice:" + str(m_f1) + " mean Jacc:" + str(m_jc) + " mean Haus:" + str(m_hf) + " mean time:" + str(m_t))
        metrics.append([{"mean iou":m_iou}, {"mean Dice": m_f1}, {"mean Jaccard": m_jc}, {"mean Haus": m_hf}, {"mean time": m_t}])
            
        # Generate metrics json file
        json_path = os.path.join(exp_folder,'metrics.json')
        with open(json_path, 'w') as f:
            f.write(json.dumps(metrics))

       

