import json
import cv2
import numpy as np
from skimage.draw import polygon2mask
import matplotlib.pyplot as plt

def reconstruct_mask(data):
    """Reconstruct the original mask from the output of an inference service.

    Args:
        data (dict): a dictionary containing the output of an inference service.

    Returns:
        mask (ndarray): a 3D Numpy array representing the reconstructed mask in RGB format.
    """

    # Get the dimensions from the output data
    width = data['images'][0]['width']
    height = data['images'][0]['height']

    # Create an empty mask with the same dimensions as the original image
    mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Define the colors for healthy and unhealthy objects (BGR format)
    color_healthy = [255, 0, 0]  # Blue
    color_unhealthy = [0, 0, 255]  # Red

    # Iterate over each segmentation
    for segmentation in data['segmentations']:
        # Each segmentation is a list of polygons, and each polygon is a list of (x, y) coordinates
        for ii, polygon in enumerate(segmentation['segmentation']):
            # The polygon coordinates are in (x, y) format, but we need them in (row, col) format (i.e., (y, x))
            polygon = [(y, x) for (x, y) in polygon]

            # Create a binary mask for this polygon using polygon2mask
            poly_mask = polygon2mask((height, width), polygon)

            # Determine the color of the object based on its health status
            color = color_healthy if segmentation['health_status'][ii] == 0 else color_unhealthy

            # Use the binary mask to set the corresponding pixels in the overall mask to the chosen color
            for i in range(3):
                mask[:, :, i] = np.where(poly_mask, color[i], mask[:, :, i])

    return mask

def apply_mask(image_path, mask):
    """
    Apply the mask to the original image.

    Args:
        image_path (str): The path to the original image.
        mask (np.ndarray): The mask to be applied.

    Returns:
        masked_image (np.ndarray): The original image with the mask applied.
    """

    # Load the original image
    image = cv2.imread(image_path)

    # Ensure the image and mask have the same size
    assert image.shape[:2] == mask.shape[:2], "Image and mask must have the same size."

    # Create a black image with the same size as the original image
    black_image = np.zeros_like(image)

    # Apply the mask to the black image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image 

def main():
    # Load the original image
    image_path = "inputs/uvas/IMG_1736.JPG"
    
    # Load the output data from the txt file
    with open('output.txt', 'r') as file:
        output_txt = file.read()

    # Primero, carga el texto en un objeto JSON
    json_obj = json.loads(output_txt)

    # Luego, puedes acceder a los valores en este objeto como un diccionario de Python
    device = json_obj["device"]
    frame = json_obj["frame"]

    # La clave "annotations_json" también es una cadena JSON, así que deberías cargarla como un objeto JSON también
    annotations = json.loads(json_obj["annotations_json"])

    # Ahora puedes acceder a los valores en "annotations" como un diccionario de Python
    info = annotations["info"]
    images = annotations["images"]
    detections = annotations["detections"]
    segmentations = annotations["segmentations"]

    # Puedes imprimir estos valores o hacer algo más con ellos
    print(f"Device: {device}")
    print(f"Frame: {frame}")
    print(f"Info: {info}")
    print(f"Images: {images}")
    print(f"Detections: {detections}")
    print(f"Segmentations: {segmentations}")

    # Call the function to reconstruct the mask
    mask = reconstruct_mask(annotations)

    # Save the mask using cv2
    cv2.imwrite('mask.png', mask)
    
    # Convert mask from RGB to grayscale (needed for cv2.bitwise_and operation)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Apply mask to the original image
    overlaid_image = apply_mask(image_path, mask)

    # Save the overlaid image
    cv2.imwrite('overlaid_image.png', overlaid_image)

if __name__ == '__main__':
    main()
