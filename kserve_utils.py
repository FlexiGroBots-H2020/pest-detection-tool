import numpy as np
import cv2
from PIL import Image
import base64
import io
import json

ENCODING = 'utf-8'

def dict2json(dictionary, output="output.json"):
    with open(output, "w") as outfile:
        json.dump(dictionary, outfile)
    return json.dumps(dictionary)
    

def encode_im_file2b64str(path):
    with open(path, "rb") as image_file:
        byte_content = image_file.read()
    base64_bytes = base64.b64encode(byte_content)
    base64_string = base64_bytes.decode(ENCODING)
    return base64_string

def encode_im_np2b64str(img_np):
    im_bytes = img_np.tobytes()
    out_img_b64_bytes = base64.b64encode(im_bytes) 
    # b64 bytes to b64 string to be serializable in json
    out_img_b64_str = out_img_b64_bytes.decode(ENCODING) 
    return out_img_b64_str

def decode_im_b642np(img_b64_str):
    img_b64_bytes = base64.b64decode(img_b64_str)
    input_image = Image.open(io.BytesIO(img_b64_bytes))
    
    # resize and change channel order
    img_np = np.array(resize(input_image, 1920))
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    return img_np

def resize(frame: Image.Image, max_res):
    f = max(*[x/max_res for x in frame.size], 1)
    if f  == 1:
        return frame
    new_shape = [int(x/f) for x in frame.size]
    return frame.resize(new_shape,  resample=Image.BILINEAR)