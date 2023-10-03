# NOT USED

# REMANE
# mask.png > imagename.png

import os
for root, dirs, names in os.walk("C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/dataset/mask_img"):   

    for dr in dirs:
        file_dir = os.path.join(root, dr)
        # print(dr)
        file = os.path.join(file_dir, '.png')
        # print(file)
        new_name = dr.split('_')[0] + '.png'
        new_file_name = os.path.join(file_dir, new_name)
        os.rename(file, new_file_name)