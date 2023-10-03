import os
import json
import cv2
import numpy as np
import random

'''
Mask visualization using labelme json format mask.
'''

def get_random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

def visualize_mask(image_path, json_path):
    # Load image
    img = cv2.imread(image_path)

    # Load Labelme JSON file
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    mask_shapes = data.get('shapes', [])

    for shape in mask_shapes:
        # Get the polygon points
        points = np.array(shape['points'], np.int32)
        points = points.reshape((-1, 1, 2))
        # print(points)

        # Get label
        label = shape.get('label', '')
        color = get_random_color()

        # Draw the polygon on the image
        cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)

        # Put the label text near the shape
        # cv2.putText(img, label, tuple(points[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Display the image with masks
    cv2.imshow('Mask Visualization', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # import image and mask
    image_path = 'dataset/img/200.png'  
    json_path = 'C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/output_labelme.json'

    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
    elif not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
    else:
        visualize_mask(image_path, json_path)


