import cv2

'''
Only crop:
crop centered at the highest point of target object
'''

def crop_centered_box(image, center, box_size):
    x, y = center
    half_size = box_size // 2
    # define the crop region
    x_min, x_max = max(x - half_size, 0), min(x + half_size, image.shape[1])
    y_min, y_max = max(y - half_size, 0), min(y + half_size, image.shape[0])
    # Crop
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image

# Loadimg
image = cv2.imread('dataset/img/2023-07-27_15-36-21_test18.png')

# Define the center point [x, y] and the box size (width and height)
center = (705, 120) # should be the lowest point!
box_size = 200  

# Crop with the centered box
cropped_image = crop_centered_box(image, center, box_size)

# Check if the cropped image has non-zero width and height
if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
    # Display the cropped image
    cv2.imshow('Cropped Image', cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Outside the image boundary")
