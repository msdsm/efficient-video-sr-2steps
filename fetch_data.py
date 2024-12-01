import pandas as pd
from PIL import Image

def fetch_imgpath(csv_path, raw_images_64_path, raw_images_128_path, hat_path):
    """
    Extracts paths for image files based on a CSV file and organizes them into a list of tuples.

    This function reads a CSV file containing image metadata, groups the images by directory, 
    and filters groups that have exactly 25 images. For each valid group, it constructs paths 
    for three input images and one ground truth (GT) image, and stores these in a list of tuples.

    Args:
        csv_path (str): Path to the CSV file containing metadata with a `directory_name` column.
        raw_images_64_path (str): Path to the directory containing 64x64 raw images.
        raw_images_128_path (str): Path to the directory containing 128x128 ground truth images.
        hat_path (str): Path to the directory containing HAT-processed images.

    Returns:
        list: A list of tuples, each containing four image paths:
            - (input_image, left_neighbor_image, right_neighbor_image, ground_truth_image)

    Example:
        >>> fetch_imgpath("metadata.csv", "./64x64/", "./128x128/", "./HAT/")
        [('HAT/example-1_HAT_SRx2_ImageNet-pretrain.jpg', 
          './64x64/example-0.jpg', 
          './64x64/example-2.jpg', 
          './128x128/example-1.jpg'),
         ...]
    """
    df = pd.read_csv(csv_path)
    lst = []
    for directory_name, group in df.groupby("directory_name"):
        count = len(group)
        if count != 25:
            continue
        for frame in range(1, 24):
            input =  hat_path + directory_name + f"-{frame}_HAT_SRx2_ImageNet-pretrain.jpg"
            input_l = raw_images_64_path + directory_name + f"-{frame-1}.jpg"
            input_r = raw_images_64_path + directory_name + f"-{frame+1}.jpg"
            gt = raw_images_128_path + directory_name + f"-{frame}.jpg"
            lst.append((input, input_l, input_r, gt))
    
    return lst