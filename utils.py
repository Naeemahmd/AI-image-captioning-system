import os
import pandas as pd


def load_dataset_path(data_dir):
    images_dir = os.path.join(data_dir, 'Images')
    captions_file = os.path.join(data_dir, 'captions.txt')
    return images_dir, captions_file

# Load captions which is like excel file.
def load_captions(captions_file):
    df = pd.read_csv(captions_file)
    captions = {}
    for index, row in df.iterrows():
        image_id, caption = row['image'], row['caption']
        if image_id not in captions:
            captions[image_id] = []
        captions[image_id].append(caption)

    incorrect_caption_counts = {img: caps for img, caps in captions.items() if len(caps) != 5}
    if incorrect_caption_counts:
        print("Images with incorrect number of captions:")
        for img_id, caps in incorrect_caption_counts.items():
            print(f"{img_id}: {len(caps)} captions")
    else:
        print("All images have exactly 5 captions.")

    # Remove images with fewer than 5 captions
    captions = {k: v for k, v in captions.items() if len(v) == 5}
    
    num_captions = sum(len(caps) for caps in captions.values())
    num_images = len(captions)
    print(f"Loaded {num_captions} captions for {num_images} images.")

    missing_data = df.isnull().sum()
    if missing_data.any():
        print("Missing data details:")
        print(missing_data)
    else:
        print("No missing data found.")
    
    return captions

# Create a list of image paths and corresponding captions
def get_image_paths_and_captions(captions, images_dir):
    image_paths = []
    captions_list = []
    missing_images = []
    for img_id, caps in captions.items():
        img_path = os.path.join(images_dir, img_id)
        if os.path.exists(img_path):
            image_paths.extend([img_path] * len(caps))
            captions_list.extend(caps)
        else:
            missing_images.append(img_id)
    
    if missing_images:
        print(f"Missing image files for {len(missing_images)} entries:")
        print(missing_images)
    else:
        print("No missing image files found.")
    
    print(f"Found {len(image_paths)} image paths and {len(captions_list)} captions.")
    return image_paths, captions_list


