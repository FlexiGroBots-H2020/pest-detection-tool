import sys
import kserve
from typing import Dict
import logging
import torch

from kserve_utils import decode_im_b642np, encode_im_np2b64str, dict2json
import torch

import sys
sys.path.insert(0, 'detectron2/')
sys.path.insert(0, 'Detic/')
sys.path.insert(0, 'Detic/third_party/CenterNet2/')


from pest_detection_kserve import load_model, init_model, infer

# constants
ENCODING = 'utf-8'
WINDOW_NAME = "Detic"

class Model(kserve.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        
        logging.info("Init kserve inference service: %s", name)
        
        logging.info("GPU available: %d" , torch.cuda.is_available())
        
        # Define and initialize all needed variables
        init_model(self)
        
        logging.info("Model loaded")
        

    def load(self):
        # Instance Detic Predictor
        try:
            load_model(self)
        except:
            logging.warning("error loading model")
        

    def predict(self, request: Dict):
        
        logging.info("Payload: %s", request)
        
        # Extract input variables from request
        img_b64_str = request["img"]
        id = request["device_id"]
        frame = request["frame_id"]
        
        try:
            im = decode_im_b642np(img_b64_str)
        except:
            logging.info("Error prepocessing image")
        
        try:  
            out_img, out_img_mask, out_mask, det_msg = infer(self, im, id, frame)
            logging.info(det_msg)
        except:
            logging.info("Error processing image")
    
        out_img_b64_str = encode_im_np2b64str(out_img)
        out_img_mask_b64_str = encode_im_np2b64str(out_img_mask)
        
        dict_out = {"device":id ,"frame":frame ,"im_detection":out_img_b64_str}
        
        #logging.info(dict_out)
        logging.info("Image processed")

        return dict2json(dict_out)


if __name__ == "__main__":
    model = Model("pest-model")
    kserve.KFServer().start([model])

