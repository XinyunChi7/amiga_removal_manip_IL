'''
Prepare Dataset for regression delta (x,y,z)

Generate delta_xyz txt file from collected xyz txt file

'''
# save from only one txt file: 

def calculate_row_differences(rows):
    differences = []
    prev_row = None

    for row in rows:
        if row and prev_row:
            diff = [b - a for a, b in zip(prev_row, row)]
            differences.append(diff)
        prev_row = row if row else prev_row

    return differences

def read_rows_from_file(file_path):
    rows = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                row = [float(num) for num in line.strip().split(',')[:3]]  # Read only the first three columns
                rows.append(row)
    return rows

def write_rows_to_file(rows, output_file):
    with open(output_file, 'w') as file:
        for row in rows:
            row_str = ', '.join([str(d) for d in row])
            file.write(row_str + '\n')

def main():
    input_file_path = 'C:/D/Imperial/Thesis/amiga_dataset/IL/xyz_dataset/eef_whisk.txt'
    output_file_path = 'C:/D/Imperial/Thesis/amiga_dataset/IL/xyz_dataset/xyz_test.txt'

    rows = read_rows_from_file(input_file_path)
    differences = calculate_row_differences(rows)
    write_rows_to_file(differences, output_file_path)

if __name__ == "__main__":
    main()


