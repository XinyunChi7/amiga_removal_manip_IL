import os
import json
import cv2
import numpy as np
import random

'''
Mask visualization with printed labels,
with given image and JSON file.
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
        points = np.array(shape['points'], np.int32)
        points = points.reshape((-1, 1, 2))

        # Get label
        label = shape.get('label', '')
        # Generate a random color 
        color = get_random_color()

        # Draw the polygon on the image
        cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)

        # Print label
        cv2.putText(img, label, tuple(points[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Display the image with masks
    cv2.imshow('Mask Visualization', img)
    # cv2.waitKey(0)
    cv2.imwrite('mask_visualization.jpg', img)
    # cv2.destroyAllWindows()


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
    # define the crop region
    x_min, x_max = max(x - half_size, 0), min(x + half_size, image.shape[1])
    y_min, y_max = max(y - half_size, 0), min(y + half_size, image.shape[0])
    # Crop the image
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image



if __name__ == '__main__':
    # import image and mask
    image_path = 'dataset/img/2023-07-27_15-48-38_test20.png'  
    json_path = 'dataset/mask/2023-07-27_15-48-38_test20.json'

    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
    elif not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
    else:
        visualize_mask(image_path, json_path) # visualize mask

    # Plot the lowest point of each mask
    masks = parse_labelme_json(json_path)
    
    image = cv2.imread('mask_visualization.jpg')
    lowest_points = []
    
    for mask in masks:
        lowest_point = find_lowest_point(mask)
        lowest_points.append(lowest_point)
        cv2.circle(image, tuple(lowest_point[::-1]), 5, (0, 255, 0), -1)
    
    cv2.imshow('Image with Lowest Points', image)
    cv2.imwrite('test.jpg', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()  

    print("Lowest points coordinates:")
    for idx, point in enumerate(lowest_points, start=1):
        print(f"Mask {idx}: X={point[1]}, Y={point[0]}")


    # Load the image
    image = cv2.imread('test.jpg')

    # Define the center point [x, y] and the box size (width and height)
    # center is the second lowest point, but switch x and y
    center = lowest_points[1][::-1]
    box_size = 250

    # Crop the image with the centered box
    cropped_image = crop_centered_box(image, center, box_size)

    # Check if the cropped image has non-zero width and height
    if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
        # Display the cropped image
        cv2.imshow('Cropped Image', cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("The crop region is outside the image boundaries.")



        