import os
import cv2 as cv
import numpy as np


def reduce_noise(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder and its sub-folder
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            file_path = os.path.join(root, file)

            # Check if the file is an image
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                try:
                    # print("Path:", file_path)
                    image = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
                    image_category = file_path.split('/')[3]

                    # Good settings: d=0, sigmalColor=10, sigmaSpace=10
                    # Also: d=0, sigmaColor=12, sigmaSpace=12
                    image_filtered = cv.bilateralFilter(image, d=0, sigmaColor=14, sigmaSpace=14)

                    # Save the result to the output folder
                    category_folder = os.path.join(output_folder, image_category)
                    if not os.path.exists(category_folder):
                        os.makedirs(category_folder)

                    output_path = os.path.join(category_folder, file)
                    cv.imwrite(output_path, image_filtered)

                    print(f'Processed: {file}')

                except Exception as e:
                    print(f'Error processing {file}: {e}')


if __name__ == '__main__':
    input_folder = '../datasets/OCTID_IMAGES'
    output_folder = '../datasets/OCTID_IMAGES_PROCESSED_BIL'
    reduce_noise(input_folder, output_folder)
