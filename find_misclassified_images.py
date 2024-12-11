import os
from collections import defaultdict

# This function reads and returns a set of misclassified image paths from a file, where each line contains information about a misclassified image.
def read_misclassified_images(file_path):
    misclassified_images = set()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                image_info = eval(line)
                image_path = image_info[0]
                misclassified_images.add(image_path)
    return misclassified_images

# This function counts the occurrences of misclassified images across multiple files and returns a dictionary with the image paths and their corresponding misclassification counts.
def count_misclassified_images(file_paths):
    image_counts = defaultdict(int)
    for file_path in file_paths:
        images = read_misclassified_images(file_path)
        for image_path in images:
            image_counts[image_path] += 1
    return image_counts

# This function returns a list of images that were misclassified by at least 'n' models, based on the counts provided in the input dictionary.
def find_images_misclassified_by_at_least_n_models(image_counts, n):
    images = [image_path for image_path, count in image_counts.items() if count >= n]
    return images

def main():
    misclassified_files = [
        '/path_to_misclassified_images_files',
    ]
    
    # check if files exist
    for file_path in misclassified_files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
    
    # count how many times the image apears
    image_counts = count_misclassified_images(misclassified_files)
    
    total_models = len(misclassified_files)
    n = total_models - 0 # n This is in all the runs, if you want less you have to reduce the quantity
   
    images_misclassified_by_at_least_n_models = find_images_misclassified_by_at_least_n_models(image_counts, n)
    
    print(f"Images misclassified by at least {n} models:")
    for image_path in images_misclassified_by_at_least_n_models:
        print(image_path)
    
    print(f"\nTotal images misclassified by at least {n} models: {len(images_misclassified_by_at_least_n_models)}")

if __name__ == "__main__":
    main()


