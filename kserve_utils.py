import numpy as np
import cv2
from PIL import Image
import base64
import io
import json
import piexif

ENCODING = 'utf-8'

def extract_metadata(exif_data):
    metadata = []
    if not exif_data or "0th" not in exif_data:
        print("Metadata not found")
        return

    specified_tags = [piexif.ImageIFD.Artist, piexif.ImageIFD.ImageID, piexif.ImageIFD.DateTime]

    for tag in specified_tags:
        tag_name = piexif.TAGS["0th"][tag]["name"]
        if tag in exif_data["0th"]:
            tag_value = exif_data["0th"][tag]
            if isinstance(tag_value, bytes):
                tag_value = tag_value.decode("utf-8")
            metadata.append(tag_value)
        else:
            print(f"{tag_name} ({tag}): not found")
    return metadata

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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
    # Encode the numpy image array to a proper image format (e.g., JPEG)
    success, img_buffer = cv2.imencode('.jpg', img_np)
    if not success:
        raise ValueError("Error encoding image")

    # Encode the image buffer to base64
    img_b64_bytes = base64.b64encode(img_buffer)
    img_b64_str = img_b64_bytes.decode(ENCODING)
    return img_b64_str

def decode_im_b642np_pil(img_b64_str):
    img_b64_bytes = base64.b64decode(img_b64_str)
    input_image = Image.open(io.BytesIO(img_b64_bytes))
    
    # resize and change channel order
    img_np = np.array(resize_pil(input_image, 1536))
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    return img_np

def resize_pil(frame: Image.Image, max_res):
    f = max(*[x/max_res for x in frame.size], 1)
    if f  == 1:
        return frame
    new_shape = [int(x/f) for x in frame.size]
    return frame.resize(new_shape,  resample=Image.BILINEAR)

def calculate_new_dimensions(image, max_dimension):
    height, width = image.shape[:2]
    aspect_ratio = float(width) / float(height)

    if height > width:
        new_height = max_dimension
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = max_dimension
        new_height = int(new_width / aspect_ratio)

    return (new_width, new_height)

def resize_image_cv2(image, new_dimensions):
    return cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

def payload2info(payload):
    # Extract input variables from request
    exif_data = piexif.load(payload)
    
    metadata = extract_metadata(exif_data)
    
    # delete metadata EXIF 
    exif_bytes = piexif.dump(exif_data)
    img_without_metadata = payload[len(exif_bytes):]
    img_data = np.frombuffer(img_without_metadata, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    return img, metadata


def decode_image(encoded_image_data):
    image_data = base64.b64decode(encoded_image_data)
    image_data = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    
    # resize and change channel order
    new_dimensions = calculate_new_dimensions(image, 1536)

    # Redimensiona la imagen
    img_np = resize_image_cv2(image, new_dimensions)
    return img_np

def ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def prepare_nested_output(nested_obj):
    output_list = []
    for item in nested_obj:
        if isinstance(item, tuple):
            new_item = tuple(ndarray_to_list(sub_item) for sub_item in item)
        else:
            new_item = ndarray_to_list(item)
        output_list.append(new_item)
    return output_list