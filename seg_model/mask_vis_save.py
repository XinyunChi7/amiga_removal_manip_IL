import os
import json
import numpy as np
from PIL import Image, ImageDraw

'''Convert mask from LabelMe annotation to a binary mask image'''

def create_mask_from_labelme(labelme_annotation, image_size):
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)

    for shape in labelme_annotation['shapes']:
        points = shape['points']
        polygon = [tuple(p) for p in points]
        draw.polygon(polygon, outline=1, fill=1)
    
    mask = np.array(mask)
    return mask


def convert_labelme_to_masks(labelme_json_dir, output_masks_dir):
    os.makedirs(output_masks_dir, exist_ok=True)

    for labelme_json_filename in os.listdir(labelme_json_dir):
        if labelme_json_filename.endswith('.json'):
            with open(os.path.join(labelme_json_dir, labelme_json_filename), 'r') as json_file:
                labelme_data = json.load(json_file)
            
            image_filename = labelme_data['imagePath']
            image_size = (labelme_data['imageWidth'], labelme_data['imageHeight'])
            mask = create_mask_from_labelme(labelme_data, image_size)

            mask_filename = os.path.splitext(image_filename)[0] + '_mask.png'
            mask_path = os.path.join(output_masks_dir, mask_filename)

            mask_image = Image.fromarray(mask * 255)
            mask_image.save(mask_path)

            # visualize the mask
            mask_visualization = Image.new('L', image_size, 0)
            mask_draw = ImageDraw.Draw(mask_visualization)
            mask_draw.rectangle([(0, 0), image_size], outline=1, width=5)
            mask_visualization = Image.blend(mask_visualization, mask_image.convert('L'), alpha=0.5)
            mask_visualization_filename = os.path.basename(mask_filename)
            mask_visualization_path = os.path.join(output_masks_dir, mask_visualization_filename)
            mask_visualization.save(mask_visualization_path)

# define directories
labelme_json_dir = 'C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/data_strainer/mask_strainer'
output_masks_dir = 'C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/data_strainer/cv2_mask'

convert_labelme_to_masks(labelme_json_dir, output_masks_dir)
