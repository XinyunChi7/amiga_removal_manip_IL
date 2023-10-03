import os
'''
RENAME IMAGES IN A FOLDER
'''
def rename_images(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    image_files.sort()

    for index, image_file in enumerate(image_files, start=1):
        extension = os.path.splitext(image_file)[1]
        # new_filename = f"{index:03d}{extension}"

        # FOR cv2_mask folder rename: rename new image file name to 'number_mask' format
        new_filename = f"{index:03d}_mask{extension}"

        old_path = os.path.join(folder_path, image_file)
        new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed {image_file} to {new_filename}")

def main():
    # folder_path = 'C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/dataset/img'
    folder_path = 'C:/D/Imperial/Thesis/amiga_dataset/IL/xyz_dataset/crop_sq_revised/img'
    rename_images(folder_path)

if __name__ == "__main__":
    main()
