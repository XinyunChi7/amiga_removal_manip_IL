import os
import json
import cv2
import numpy as np
import random

'''
Mask visualization & SAVE processed imgs:
Visualize masks and save cropped images with background information
'''

# Define color map
MASK_COLOR_MAP = {
    "whisk": (0, 255, 0),   # Green
    "hook": (255, 0, 0),   # Blue
    # Add more mask types and colors
}

def visualize_mask(image_path, json_path, output_folder):
    # Load image
    img = cv2.imread(image_path)

    # Load Labelme JSON file
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    # Retrieve mask annotations from the JSON
    mask_shapes = data.get('shapes', [])
    # Assign color for each mask 
    mask_colors = {}

    for shape in mask_shapes:
        # Get the label 
        label = shape.get('label', '')
        print(label)

        # Assigin color for each mask type
        color = MASK_COLOR_MAP.get(label, (0, 0, 0))  # Default to black if label not found
        if label not in mask_colors:
            mask_colors[label] = color
            
        # Merge colored masks with black background
        points = np.array(shape['points'], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)

    # Save the image with masks
    output_image_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_image_path, img)

def process_images(input_image_folder, mask_folder, output_folder):
    image_files = os.listdir(input_image_folder)
    
    for image_file in image_files:
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_image_folder, image_file)
            json_file = os.path.splitext(image_file)[0] + '.json'
            json_path = os.path.join(mask_folder, json_file)
            
            if os.path.exists(json_path):
                visualize_mask(image_path, json_path, output_folder)
                print(f"Processed: {image_file}")
            else:
                print(f"JSON file not found for: {image_file}")



def parse_labelme_json(labelme_json_path):
    with open(labelme_json_path, 'r') as json_file:
        data = json.load(json_file)
    shapes = data['shapes']
    masks = []
    for shape in shapes:
        if shape['shape_type'] == 'polygon':
            mask = np.zeros(data['imageHeight'] * data['imageWidth'], dtype=np.uint8)
            points = shape['points']
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask.reshape(data['imageHeight'], data['imageWidth']), [points], 1)
            masks.append(mask.reshape(data['imageHeight'], data['imageWidth']))
    return masks

def find_lowest_point(mask):
    nonzero_points = np.argwhere(mask)
    lowest_point = nonzero_points[np.argmin(nonzero_points[:, 0])]
    print(lowest_point)
    return lowest_point

def crop_centered_box(image, center, box_size):
    x, y = center
    half_size = box_size // 2
    x_min, x_max = max(x - half_size, 0), min(x + half_size, image.shape[1])
    y_min, y_max = max(y - half_size, 0), min(y + half_size, image.shape[0])
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image


if __name__ == '__main__':

    input_image_folder = 'C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/dataset/img'
    mask_folder = 'C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/dataset/mask'
    output_folder = 'C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/dataset/crop_mask_img'
    
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    process_images(input_image_folder, mask_folder, output_folder)
    
    for output_image_file in os.listdir(output_folder):
        if output_image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            output_image_path = os.path.join(output_folder, output_image_file)
            image = cv2.imread(output_image_path)

            json_file = os.path.splitext(output_image_file)[0] + '.json'
            json_path = os.path.join(mask_folder, json_file)
            masks = parse_labelme_json(json_path)
            
            lowest_points = []
            for mask in masks:
                lowest_point = find_lowest_point(mask)
                lowest_points.append(lowest_point)
                # cv2.circle(image, tuple(lowest_point[::-1]), 5, (0, 255, 0), -1)
            
            if len(lowest_points) > 1:
                center = lowest_points[1][::-1]  # Use the second lowest point for cropping
                box_size = 250
                cropped_image = crop_centered_box(image, center, box_size)

                if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
                    cropped_output_path = os.path.join(output_folder, 'cropped_' + output_image_file)
                    cv2.imwrite(cropped_output_path, cropped_image)
                    print(f"Cropped and saved: {cropped_output_path}")
                else:
                    print("The crop region is outside the image boundaries.")
            else:
                print("Insufficient lowest points for cropping.")


