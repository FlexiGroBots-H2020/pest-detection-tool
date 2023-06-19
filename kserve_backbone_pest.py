import sys
import kserve
from typing import Dict
import logging
import torch

import time
import json
import os

from kserve_utils import payload2info

import sys
sys.path.insert(0, 'detectron2/')
sys.path.insert(0, 'Detic/')
sys.path.insert(0, 'Detic/third_party/CenterNet2/')
import paho.mqtt.publish as publish


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
        try:
            init_model(self)
            logging.info("Model initialized")
        except Exception as e:
            logging.warning("error init model: " + str(e))
        
        # Instance models
        try:
            load_model(self)
        except Exception as e:
            logging.warning("error loading models: {}".format(e))
        
        logging.info("Models loaded")
        

    def predict(self, request: Dict):
        
        logging.info("Predict call -------------------------------------------------------")
        
        #logging.info("Payload: %s", request)
        start_time = time.time()
        try:
            img, metadata = payload2info(request)
            id = metadata[0]
            frame_id = metadata[1]
            init_time = metadata[2]
            logging.info("Payload image shape: {}, device: {}, frame: {}".format(img.shape, id, frame_id))
        except Exception as e:
            logging.info("Error prepocessing image: {}".format(e))
        
        decode_time = time.time() - start_time
        logging.info(f"Im Decode and metadata extracted in time: {decode_time:.2f}s")
        
        try:  
            annotations_json, cont_det = infer(self, img, id, frame_id)
            logging.info("Num detections: {}".format(cont_det))
        except Exception as e:
            logging.info("Error processing image: {}".format(e))
        
        #logging.info(dict_out)
        logging.info("Image processed")
        # Encode out imgs
        start_time = time.time()
        dict_out = {"device":id ,"frame":frame_id, "init_time": init_time , "annotations_json": annotations_json}
        encode_time = time.time() - start_time
        logging.info(f"dict out time: {encode_time:.4f}s")
        logging.info("Image processed")
        
        # Publish a message 
        start_time = time.time()
        mqtt_topic = "common-apps/pest-model/output/" + id
        client_id = self.name + "_" + id
        publish.single(mqtt_topic, 
                       json.dumps(dict_out), 
                       hostname=os.getenv('BROKER_ADDRESS'), 
                       port=int(os.getenv('BROKER_PORT')), 
                       client_id=client_id, 
                       auth = {"username": os.getenv('BROKER_USER'), "password": os.getenv('BROKER_PASSWORD')} )
        encode_time = time.time() - start_time
        logging.info(f"Publish out time: {encode_time:.2f}s")

        return {}


if __name__ == "__main__":
    model = Model("pest-model")
    kserve.KFServer().start([model])

