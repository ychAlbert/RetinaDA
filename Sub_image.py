import numpy as np  
import os  
import random 
import cv2  
from PIL import Image 
import hashlib  

# Control the number of executions and other parameters
execution_count = 1  # Number of times to perform image processing
current_index = 4  # Current index, used for naming the output directory
photo_index = 1  # Used for naming the output files

# Function to convert GIF/TIF to PNG
# Input : image_path (path of the input image), output_dir (path of the output directory)
# Output : png_path (path of the converted image)
def convert_to_png(image_path, output_dir):
    # Open the image file
    img = Image.open(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]  # Get the file name without the extension
    file_name = base_name
    
    # Save as PNG format
    png_path = os.path.join(output_dir, f"{file_name}.png")
    img.save(png_path, 'PNG')
    print(f"Converted {os.path.splitext(os.path.basename(image_path))[1]} to PNG: {png_path}")
    
    return png_path

# Function to check if the image is in GIF or TIF format
# Input : image_path (path of the input image)
# Output : True if the image is in GIF or TIF format, False otherwise
def is_convertible(image_path):
    return image_path.lower().endswith(('.gif', '.tif'))

# Image processing function
# Input : image_path (path of the input image), output_dir (path of the output directory), center (coordinates of the center of the crop), seed (random seed), size_range (minimum and maximum size for cropping images)
# Output : None
def process_image(image_path, output_dir, center, seed, size_range):
    random.seed(seed)  # Initialize the random number generator with a seed
    # Preprocessing: Load the image as a grayscale image, find the positions of the minimum and maximum values
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(image)

    # Load the image as a color image, convert to RGB format
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Add rotation operation
    photo_angle = random.randint(0, 360)
    rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), photo_angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    # Randomly choose whether to flip the image
    photo_fz = random.randint(0, 1)
    if photo_fz:
        flipped_image = cv2.flip(rotated_image, 0)  # Flip along the vertical axis
    else:
        flipped_image = rotated_image

    # Cropping based on coordinates
    center_x, center_y = center
    size = random.randint(size_range[0], size_range[1])  # Randomly determine the size of the crop
    center_x = center_x + random.randint(-150, 150)  # Randomly offset the crop center point
    center_y = center_y + random.randint(-150, 150)  # Randomly offset the crop center point

    start_x = max(min(center_x - size // 2, image.shape[1] - size), 0)
    start_y = max(min(center_y - size // 2, image.shape[0] - size), 0)
    end_x = min(center_x + size // 2, image.shape[1])
    end_y = min(center_y + size // 2, image.shape[0])

    cropped_image = flipped_image[start_y:end_y, start_x:end_x]

    # Data normalization
    cropped_image = ((cropped_image - np.min(cropped_image)) / (np.max(cropped_image) - np.min(cropped_image)) * 255).astype(np.uint8)
    resized_image = cv2.resize(cropped_image, (512, 512), interpolation=cv2.INTER_AREA)

    # Save the cropped image
    base_name = os.path.splitext(os.path.basename(image_path))[0]  # Get the file name without the extension
    output_path = os.path.join(output_dir, f"{base_name}_{photo_index}.png")
    Image.fromarray(resized_image).save(output_path)
    print(f"Cropped image saved to: {output_path}")

# Input folder paths
input_dir1 = 'YourImagePath'  # First input directory containing the images to be processed. The  images in this directory will be use to find the center point
input_dir2 = 'YourImagePath2'  # Second input directory

# Initial output folder paths
output_dir1 = f'YourImagePath\YourImageName_{current_index}'  # First output directory
output_dir2 = f'YourImagePath\YourImageName_{current_index}'  # Second output directory

# Initial crop size range
size_range = [1000, 1100]  # Minimum and maximum size for cropping images

# Ensure output directories exist
if not os.path.exists(output_dir1):
    os.makedirs(output_dir1)
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)

# Function to get the file name without the extension
def get_file_name_without_extension(file_name):
    return os.path.splitext(file_name)[0]

# Build a dictionary of file names corresponding to file names
files1 = {get_file_name_without_extension(f): f for f in os.listdir(input_dir1) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.gif'))}
files2 = {get_file_name_without_extension(f): f for f in os.listdir(input_dir2) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.gif'))}

# Ensure that the file names in both folders are the same (without extension)
common_files = set(files1.keys()).intersection(files2.keys())

# Store center coordinates and random seeds
centers = {}

# Calculate the center coordinates of the image
for file_name in common_files:
    image_path1 = os.path.join(input_dir1, files1[file_name])
    
    # If it's a GIF or TIF file, convert it to PNG first
    if is_convertible(image_path1):
        image_path1 = convert_to_png(image_path1, output_dir1)

    image = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(image)
    x, y = max_loc[0], max_loc[1]
    centers[file_name] = (x, y)

# Perform image processing
for iteration in range(execution_count):
    for file_name in common_files:
        seed = hash(file_name)  # Use the hash of the file name as the seed
        image_path1 = os.path.join(input_dir1, files1[file_name])
        image_path2 = os.path.join(input_dir2, files2[file_name])
        
        # If it's a GIF or TIF file, convert it to PNG first
        if is_convertible(image_path1):
            image_path1 = convert_to_png(image_path1, output_dir1)
        if is_convertible(image_path2):
            image_path2 = convert_to_png(image_path2, output_dir2)

        center = centers[file_name]
        for i in range(20):
            center = centers[file_name]
            seed = hash(file_name) + i  # Increase the random seed
            process_image(image_path1, output_dir1, center, seed, size_range)
            process_image(image_path2, output_dir2, center, seed, size_range)
            photo_index += 1
        photo_index = 1  # Reset photo_index
    # Update the output directory
    current_index += 1
    output_dir1 = f'YourImagePath\YourImageName_{current_index}'
    output_dir2 = f'YourImagePath\YourImageName_{current_index}'
    
    # Ensure output directories exist
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    if not os.path.exists(output_dir2):
        os.makedirs(output_dir2)
    
    # Update the crop size range
    size_range[0] += 50
    size_range[1] += 50 
