# Copyright (c) Facebook, Inc. and its affiliates.
import glob
import multiprocessing as mp
import shutil

import numpy as np
import os
import time
import cv2
import tqdm
from pest_detection_utils import non_max_suppression, setup_cfg, get_parser, det2img


import sys
sys.path.insert(0, 'detectron2/')
sys.path.insert(0, 'Detic/')
sys.path.insert(0, 'Detic/third_party/CenterNet2/')

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from Detic.detic.predictor import VisualizationDemo


# constants
WINDOW_NAME = "Detic"

def detic_single_im(args):
    if len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
    for path in tqdm.tqdm(args.input, disable=not args.output):
        img = read_image(path, format="BGR")
        start_time = time.time()
        predictions, visualized_output = detic_predictor.run_on_image(img)
        
        predictions_nms = non_max_suppression(predictions, iou_thres=args.nms_max_overlap)
        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions_nms["instances"]))
                if "instances" in predictions_nms
                else "finished",
                time.time() - start_time,
            )
        )
        
        classes_names = detic_predictor.metadata.thing_classes
        
        if len(predictions_nms['instances']) != 0:
            img_out_bbox, img_out_mask, mask_tot = det2img(img, predictions_nms, classes_names)
        else: 
            img_out_bbox = img
            img_out_mask = img
            mask_tot = np.zeros((img.shape[0], img.shape[1]))
        
        if args.output:
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, os.path.basename(path))
            else:
                assert len(args.input) == 1, "Please specify a directory with args.output"
                out_filename = args.output
        else:
            out_filename=""
        
        cv2.imwrite(out_filename +"_vis.jpg", img_out_bbox)
        cv2.imwrite(out_filename +"_mask.jpg", img_out_mask)
            
def detic_video(args):
    # set video input parameters
    video = cv2.VideoCapture(args.video_input)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    basename = os.path.basename(args.video_input)

    # Create Videowriters to generate video output
    file_ext = ".avi"
    path_out_vis = os.path.join(args.output, basename.split(".")[0] + file_ext)
    path_out_mask = os.path.join(args.output, basename.split(".")[0] + "_mask" + file_ext)
    output_file_vis = cv2.VideoWriter(path_out_vis, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                                        (width, height))
    output_file_mask = cv2.VideoWriter(path_out_mask, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                                        (width, height))

    frame_count = 0
    # Processing loop
    while (video.isOpened()):
        # read frame
        ret, frame = video.read()
        if frame is None:
            break
        # predict detections DETIC
        start_time = time.time()
        predictions, visualized_output = detic_predictor.run_on_image(frame)
        predictions_nms = non_max_suppression(predictions, iou_thres=args.nms_max_overlap)
        logger.info(
            "{}: {} in {:.2f}s".format(
                frame_count,
                "detected {} instances".format(len(predictions_nms["instances"]))
                if "instances" in predictions_nms
                else "finished",
                time.time() - start_time,
            )
        )
    
        classes_names = detic_predictor.metadata.thing_classes
        
        if len(predictions_nms['instances']) != 0:
            img_out_bbox, img_out_mask, mask_tot = det2img(frame, predictions_nms, classes_names)
        else: 
            img_out_bbox = frame
            img_out_mask = frame
            mask_tot = np.zeros((frame.shape[0], frame.shape[1]))
            
        # write results to video output
        output_file_vis.write(np.uint8(img_out_bbox))
        output_file_mask.write(np.uint8(img_out_mask))
        frame_count = frame_count + 1
        end_time = time.time() - start_time
        print("Detection finished in " + str(round(end_time, 2)) + "s")

        
    # Release VideoCapture and VideoWriters
    video.release()
    output_file_vis.release()        


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    if args.debug:
        if not os.path.exists(args.debug):
            os.mkdir(args.debug)
        else:
            shutil.rmtree(args.debug)
            os.mkdir(args.debug)

    # Instance Detic Predictor
    try:
        detic_predictor = VisualizationDemo(cfg, args)
    except:
        # second time it works
        detic_predictor = VisualizationDemo(cfg, args)
        logger.warning("w: 'CustomRCNN' was already registered")

    frame_count = 0

    # Infer in single images
    if args.input:
        detic_single_im(args)

    # Infer in video
    elif args.video_input:
        detic_video(args)
        