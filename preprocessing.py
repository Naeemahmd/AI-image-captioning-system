import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def load_dataset_path(data_dir):
    images_dir = os.path.join(data_dir, 'Images')
    captions_file = os.path.join(data_dir, 'captions.txt')
    return images_dir, captions_file

# Load captions which is like excel file. also limit number of samlpes if needed.
def load_captions(captions_file, num_samples=10):
    df = pd.read_csv(captions_file)
    captions = {}
    for index, row in df.iterrows():
        image_id, caption = row['image'], row['caption']
        if image_id not in captions:
            captions[image_id] = []
        captions[image_id].append(caption)
        # Stop after loading the first 1000 samples
        if len(captions) >= num_samples:
            break

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

class TokenizerWrapper:
    """
    Wrapper class for Keras Tokenizer to handle text tokenization and padding.
    """
    def __init__(self, num_words=5000):
        self.tokenizer = Tokenizer(num_words=num_words)
        self.max_length = None

    def fit_on_texts(self, captions_list):
        self.tokenizer.fit_on_texts(captions_list)
        sequences = self.tokenizer.texts_to_sequences(captions_list)
        self.max_length = max(len(seq) for seq in sequences)

    def texts_to_sequences(self, text):
        return self.tokenizer.texts_to_sequences([text])[0]

    def pad_sequences(self, sequences):
        return pad_sequences([sequences], maxlen=self.max_length)[0]

    def get_vocab_size(self):
        return len(self.tokenizer.word_index) + 1

    def get_max_length(self):
        return self.max_length


# Extract features using ResNet50
def extract_features(image_paths):
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    features = {}
    for img_path in image_paths:
        image = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = tf.keras.applications.resnet50.preprocess_input(image)
        feature = model.predict(image, verbose=0)
        img_id = os.path.basename(img_path).split('.')[0]
        features[img_id] = feature.flatten()  # Ensure features are flattened to shape (2048,)
    return features

def create_sequences(tokenizer, max_length, captions_list, image_features, vocab_size, image_paths):
    X1, X2, y = [], [], []
    for i, desc in enumerate(captions_list):
        seq = tokenizer.texts_to_sequences([desc])
        if len(seq) == 0 or len(seq[0]) == 0:
            print(f"Skipping empty sequence for caption: {desc}")
            continue
        seq = seq[0]
        for j in range(1, len(seq)):
            in_seq, out_seq = seq[:j], seq[j]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)
            img_id = os.path.basename(image_paths[i // 5]).split('.')[0]
            if img_id in image_features:
                X1.append(image_features[img_id])  # Use flattened feature
                X2.append(in_seq)
                y_seq = np.zeros((max_length, vocab_size))
                y_seq[j - 1] = out_seq
                y.append(y_seq)
            else:
                print(f"Missing image feature for: {img_id}")
    return np.array(X1), np.array(X2), np.array(y)


def split_to_train_test(captions, test_size=0.1):
    """" 
    Split the captions dictionary into training and validation sets.

    """
    image_ids = list(captions.keys())
    train_ids, val_ids = train_test_split(image_ids, test_size=test_size, random_state=42)
    
    train_captions = {img_id: captions[img_id] for img_id in train_ids}
    val_captions = {img_id: captions[img_id] for img_id in val_ids}
    
    return train_captions, val_captions