'''
Prepare Dataset for regression delta (x,y,z)

Generate delta_xyz txt file from collected xyz txt file

'''

# save from multiple txt files:
# if you want to save from one singel txt file, check 'reg_data_prep_single.py'

def calculate_row_differences(numbers):
    differences = []
    for i in range(1, len(numbers)):
        diff = [b - a for a, b in zip(numbers[i - 1], numbers[i])]
        differences.append(diff)
    return differences

def read_numbers_from_file(file_path):
    numbers = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [float(num) for num in line.strip().split(',')[:3]]  # Read first three columns (xyz)
            numbers.append(row)
    return numbers

def write_differences_to_file(differences, output_file):
    with open(output_file, 'w') as file:
        for diff_list in differences:
            diff_str = ', '.join([str(d) for d in diff_list])
            file.write(diff_str + '\n')


def main():
    input_file_paths = ['C:/D/Imperial/Thesis/amiga_dataset/cv_eef_dataset/eef_test/eff_whisk/eef_trail_18.txt', 
                        'C:/D/Imperial/Thesis/amiga_dataset/cv_eef_dataset/eef_test/eff_whisk/eef_trail_19.txt',
                        'C:/D/Imperial/Thesis/amiga_dataset/cv_eef_dataset/eef_test/eff_whisk/eef_trail_20.txt',
                        'C:/D/Imperial/Thesis/amiga_dataset/cv_eef_dataset/eef_test/eff_whisk/eef_trail_21.txt',
                        'C:/D/Imperial/Thesis/amiga_dataset/cv_eef_dataset/eef_test/eff_whisk/eef_trail_22.txt',
                        'C:/D/Imperial/Thesis/amiga_dataset/cv_eef_dataset/eef_test/eff_whisk/eef_trail_23.txt',
                        'C:/D/Imperial/Thesis/amiga_dataset/cv_eef_dataset/eef_test/eff_whisk/eef_trail_24.txt',
                        'C:/D/Imperial/Thesis/amiga_dataset/cv_eef_dataset/eef_test/eff_whisk/eef_trail_25.txt',
                        'C:/D/Imperial/Thesis/amiga_dataset/cv_eef_dataset/eef_test/eff_whisk/eef_trail_26.txt',
                        'C:/D/Imperial/Thesis/amiga_dataset/cv_eef_dataset/eef_test/eff_whisk/eef_trail_27.txt',
                        'C:/D/Imperial/Thesis/amiga_dataset/cv_eef_dataset/eef_test/eff_whisk/eef_trail_28.txt',
                        'C:/D/Imperial/Thesis/amiga_dataset/cv_eef_dataset/eef_test/eff_whisk/eef_trail_29.txt',
                        'C:/D/Imperial/Thesis/amiga_dataset/cv_eef_dataset/eef_test/eff_whisk/eef_trail_30.txt'] 
    
    
    output_file_path = 'C:/D/Imperial/Thesis/amiga_dataset/IL/xyz_dataset/reg_test.txt'
    all_differences = []

    for input_file_path in input_file_paths:
        numbers = read_numbers_from_file(input_file_path)
        differences = calculate_row_differences(numbers)
        all_differences.extend(differences)  # Save differences for each file

    write_differences_to_file(all_differences, output_file_path)
    print(f"All differences calculated and saved to {output_file_path}")


if __name__ == "__main__":
    main()

