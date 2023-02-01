import numpy as np
import sys
from typing import Dict
import logging
import multiprocessing as mp
import numpy as np
import time
from pest_detection_utils import non_max_suppression, setup_cfg, get_parser, det2img
import sys

sys.path.insert(0, 'detectron2/')
sys.path.insert(0, 'Detic/')
sys.path.insert(0, 'Detic/third_party/CenterNet2/')

from Detic.detic.predictor import VisualizationDemo

# constants
ENCODING = 'utf-8'
WINDOW_NAME = "Detic"

def init_model(self):
     # Define and initialize all needed variables
    self.detic_predictor = None
    mp.set_start_method("spawn", force=True)
    self.args = get_parser().parse_args()
    

    self.cfg = setup_cfg(self.args)
    
    # Load the model
    logging.info("Load model")
    self.load()
    
    
def load_model(self):
    # Instance Detic Predictor
    try:
        self.detic_predictor = VisualizationDemo(self.cfg, self.args)
    except:
        # second time it works
        self.detic_predictor = VisualizationDemo(self.cfg, self.args)


def infer(self,img, id=0, frame=0):
    # img in numpy format
    start_time = time.time()
    predictions, visualized_output = self.detic_predictor.run_on_image(img)
    predictions_nms = non_max_suppression(predictions, iou_thres=self.args.nms_max_overlap)
    det_msg = "{}: {} in {:.2f}s".format(
                str(id) + "_" + str(frame),
                "detected {} instances".format(len(predictions_nms["instances"]))
                if "instances" in predictions_nms
                else "finished",
                time.time() - start_time,
            )
    classes_names = self.detic_predictor.metadata.thing_classes
        
    if len(predictions_nms['instances']) != 0:
        img_out_bbox, img_out_mask, mask_tot = det2img(img, predictions_nms, classes_names)
    else: 
        img_out_bbox = img
        img_out_mask = img
        mask_tot = np.zeros((img.shape[0], img.shape[1]))
        
    return img_out_bbox, img_out_mask, mask_tot, det_msg
                