from shutil import copyfile
import os
import shutil
import random

# Prepare Dataset for state classification
# Randomly Split into Testing 10% and Training 90%

try:
    os.mkdir('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/training')
    os.mkdir('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/testing')

    os.mkdir('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/training/good')
    os.mkdir('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/training/too_left')
    os.mkdir('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/training/too_right')
    os.mkdir('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/training/too_far')
    os.mkdir('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/training/too_high')
    os.mkdir('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/training/too_low')

    os.mkdir('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/testing/good')
    os.mkdir('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/testing/too_left')
    os.mkdir('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/testing/too_right')
    os.mkdir('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/testing/too_far')
    os.mkdir('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/testing/too_high')
    os.mkdir('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/testing/too_low')

except OSError:
    pass

# Make sure file size larger than zero, not empty
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, ignore it!")

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[-testing_length:]

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)
    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)


GOOD_SOURCE_DIR = "C:/D/Imperial/Thesis/amiga_dataset/cv_eef_dataset/imgs_test/whisk_with_hook/IL_test/good/"
TRAINING_GOOD_DIR = "C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/training/good/"
TESTING_GOOD_DIR = "C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/testing/good/"

left_SOURCE_DIR = "C:/D/Imperial/Thesis/amiga_dataset/cv_eef_dataset/imgs_test/whisk_with_hook/IL_test/too_left/"
TRAINING_left_DIR = "C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/training/too_left/"
TESTING_left_DIR = "C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/testing/too_left/"

right_SOURCE_DIR = "C:/D/Imperial/Thesis/amiga_dataset/cv_eef_dataset/imgs_test/whisk_with_hook/IL_test/too_right/"
TRAINING_right_DIR = "C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/training/too_right/"
TESTING_right_DIR = "C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/testing/too_right/"

far_SOURCE_DIR = "C:/D/Imperial/Thesis/amiga_dataset/cv_eef_dataset/imgs_test/whisk_with_hook/IL_test/too_far/"
TRAINING_far_DIR = "C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/training/too_far/"
TESTING_far_DIR = "C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/testing/too_far/"

high_SOURCE_DIR = "C:/D/Imperial/Thesis/amiga_dataset/cv_eef_dataset/imgs_test/whisk_with_hook/IL_test/too_high/"
TRAINING_high_DIR = "C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/training/too_high/"
TESTING_high_DIR = "C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/testing/too_high/"

low_SOURCE_DIR = "C:/D/Imperial/Thesis/amiga_dataset/cv_eef_dataset/imgs_test/whisk_with_hook/IL_test/too_low/"
TRAINING_low_DIR = "C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/training/too_low/"
TESTING_low_DIR = "C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/testing/too_low/"



def create_dir(file_dir):
    if os.path.exists(file_dir):
        print('true')
        shutil.rmtree(file_dir)
        os.makedirs(file_dir)
    else:
        os.makedirs(file_dir)

create_dir(TRAINING_GOOD_DIR)
create_dir(TESTING_GOOD_DIR)
create_dir(TRAINING_left_DIR)
create_dir(TESTING_left_DIR)
create_dir(TRAINING_right_DIR)
create_dir(TESTING_right_DIR)
create_dir(TRAINING_far_DIR)
create_dir(TESTING_far_DIR)
create_dir(TRAINING_high_DIR)
create_dir(TESTING_high_DIR)
create_dir(TRAINING_low_DIR)
create_dir(TESTING_low_DIR)

# Define split size (0.9 by default)
split_size = 0.9
split_data(GOOD_SOURCE_DIR, TRAINING_GOOD_DIR, TESTING_GOOD_DIR, split_size)
split_data(left_SOURCE_DIR, TRAINING_left_DIR, TESTING_left_DIR, split_size)
split_data(right_SOURCE_DIR, TRAINING_right_DIR, TESTING_right_DIR, split_size)
split_data(far_SOURCE_DIR, TRAINING_far_DIR, TESTING_far_DIR, split_size)
split_data(high_SOURCE_DIR, TRAINING_high_DIR, TESTING_high_DIR, split_size)
split_data(low_SOURCE_DIR, TRAINING_low_DIR, TESTING_low_DIR, split_size)

# Check size of training and test datasets
print("training 'good': ", len(os.listdir('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/training/good/')))
print("training 'too_left': ", len(os.listdir('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/training/too_left/')))
print("training 'too_right': ", len(os.listdir('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/training/too_right/')))
print("training 'too_far': ", len(os.listdir('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/training/too_far/')))
print("training 'too_high': ", len(os.listdir('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/training/too_high/')))
print("training 'too_low': ", len(os.listdir('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/training/too_low/')))

print("testing 'good': ", len(os.listdir('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/testing/good/')))
print("testing 'too_left': ", len(os.listdir('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/testing/too_left/')))
print("testing 'too_right': ", len(os.listdir('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/testing/too_right/')))
print("testing 'too_far': ", len(os.listdir('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/testing/too_far/')))
print("testing 'too_high': ", len(os.listdir('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/testing/too_high/')))
print("testing 'too_low': ", len(os.listdir('C:/D/Imperial/Thesis/amiga_dataset/IL/dataset/testing/too_low/')))

# Print dataset size
# print('shape of train data is ',train_data.shape)
# print('shape of train labels is ',train_labels.shape)
# print('an example of train data is ',train_data[5])

