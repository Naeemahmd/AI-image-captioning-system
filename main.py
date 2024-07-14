from utils import load_dataset_path, load_captions, get_image_paths_and_captions
from cnn import tokenize_captions, extract_features

# load images and labelled captions
data_path = "archive/"
images_dir, captions_file = load_dataset_path(data_path)
captions = load_captions(captions_file)
image_paths, captions_list = get_image_paths_and_captions(captions, images_dir)
tokenizer, captions_padded, max_length = tokenize_captions(captions_list)
image_features = extract_features(image_paths)
    

print("Done")