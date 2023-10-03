import os
import cv2
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import random

'''
Mask visualization using coco format mask.'''


def visualize_mask(coco, image_id, img_dir):
    img = coco.imgs[image_id]
    image = np.array(Image.open(os.path.join(img_dir)))

    cat_ids = coco.getCatIds()
    anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(anns_ids)

    # Create copy
    image_with_mask = image.copy()

    # Colors for the annotations
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(anns))]

    # Overlay the masks with different colors for each annotation
    for i, ann in enumerate(anns):
        mask = coco.annToMask(ann)
        color = colors[i]
        mask_binary = mask > 0
        overlay = np.zeros_like(image)
        overlay[mask_binary] = color

        # Combine the mask overlay with the original image
        image_with_mask = cv2.addWeighted(image_with_mask, 1, overlay, 0.5, 0)

    # Display the image with the mask overlay
    plt.figure(figsize=(10, 10))
    plt.imshow(image_with_mask)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    coco = COCO('C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/predicted_mask_1.json')
    img_dir = 'C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/dataset/img/022.png'

    image_id = 1

    visualize_mask(coco, image_id, img_dir)
