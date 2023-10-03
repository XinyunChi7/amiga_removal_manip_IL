import os
import shutil

'''
Split data (images and corresponding delta xyz) into training and test sets
'''

def divide_data(input_image_folder, input_txt_file, output_train_folder, output_test_folder, split_ratio=0.8):
    if not (0 <= split_ratio <= 1):
        print("Invalid split ratio, (0,1)")
        return

    if not os.path.exists(output_train_folder):
        os.makedirs(output_train_folder)
    if not os.path.exists(output_test_folder):
        os.makedirs(output_test_folder)

    image_files = os.listdir(input_image_folder)
    image_files.sort()

    split_index = int(len(image_files) * split_ratio)
    train_image_files = image_files[:split_index]
    test_image_files = image_files[split_index:] # test set is the last 10% of the data (Original order!)

    # Copy image files
    for filename in train_image_files:
        src_image_path = os.path.join(input_image_folder, filename)
        dest_image_path = os.path.join(output_train_folder, filename)
        shutil.copy(src_image_path, dest_image_path)

    for filename in test_image_files:
        src_image_path = os.path.join(input_image_folder, filename)
        dest_image_path = os.path.join(output_test_folder, filename)
        shutil.copy(src_image_path, dest_image_path)

    # Process the txt file
    with open(input_txt_file, 'r') as txt_file:
        lines = txt_file.readlines()

    train_lines = lines[:split_index]
    test_lines = lines[split_index:]

    train_txt_file = os.path.join(output_train_folder, 'targets.txt')
    test_txt_file = os.path.join(output_test_folder, 'targets.txt')

    with open(train_txt_file, 'w') as train_file:
        train_file.writelines(train_lines)

    with open(test_txt_file, 'w') as test_file:
        test_file.writelines(test_lines)

def main():
#     input_image_folder = 'C:/D/Imperial/Thesis/amiga_dataset/IL/xyz_dataset/images'
#     input_txt_file = 'C:/D/Imperial/Thesis/amiga_dataset/IL/xyz_dataset/reg_test.txt'
    
    input_image_folder = 'C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/all_data/crop_sq_0409/img'
    input_txt_file = 'C:/D/Imperial/Thesis/amiga_dataset/SAM_test/seg_test/all_data/crop_sq_0409/reg_test_0904.txt'

    output_train_folder = 'C:/D/Imperial/Thesis/amiga_dataset/IL/act_pred_0904/train'
    output_test_folder = 'C:/D/Imperial/Thesis/amiga_dataset/IL/act_pred_0904/test'
    split_ratio = 0.9

    divide_data(input_image_folder, input_txt_file, output_train_folder, output_test_folder, split_ratio)
    print("Data and targets divided into training and test sets.")

if __name__ == "__main__":
    main()


# def divide_data(input_folder, output_train_folder, output_test_folder, split_ratio=0.8):
#     if not (0 <= split_ratio <= 1):
#         print("Invalid split ratio. Please use a value between 0 and 1.")
#         return

#     if not os.path.exists(output_train_folder):
#         os.makedirs(output_train_folder)
#     if not os.path.exists(output_test_folder):
#         os.makedirs(output_test_folder)

#     file_list = os.listdir(input_folder)
#     file_list.sort()

#     split_index = int(len(file_list) * split_ratio)
#     train_files = file_list[:split_index]
#     test_files = file_list[split_index:]

#     for filename in train_files:
#         src_path = os.path.join(input_folder, filename)
#         dest_path = os.path.join(output_train_folder, filename)
#         shutil.copy(src_path, dest_path)

#     for filename in test_files:
#         src_path = os.path.join(input_folder, filename)
#         dest_path = os.path.join(output_test_folder, filename)
#         shutil.copy(src_path, dest_path)

# def main():
#     input_folder = 'C:/D/Imperial/Thesis/amiga_dataset/IL/xyz_dataset/images'
#     output_train_folder = 'C:/D/Imperial/Thesis/amiga_dataset/IL/act_pred_dataset/train'
#     output_test_folder = 'C:/D/Imperial/Thesis/amiga_dataset/IL/act_pred_dataset/test'
#     split_ratio = 0.8

#     divide_data(input_folder, output_train_folder, output_test_folder, split_ratio)
#     print("Data divided into training and test sets.")

# if __name__ == "__main__":
#     main()
